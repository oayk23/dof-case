import pandas as pd
import numpy as np
import faiss
import pickle
import os
from typing import List, Tuple, Dict
from transformers import AutoModel,AutoTokenizer
import logging
import torch
from torch.nn import functional as F
from pathlib import Path
import tiktoken
# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def tokenize_and_chunk_corpus(corpus_text: str, 
                             chunk_size: int = 500, 
                             overlap_percentage: float = 0.1,
                             output_filename: str = "chunked_corpus.csv") -> pd.DataFrame:
    """
    Corpus'u tokenlaştırır, belirtilen boyutta chunk'lara ayırır ve CSV olarak kaydeder.
    
    Args:
        corpus_text: İşlenecek metin corpus'u
        chunk_size: Her chunk'ın token sayısı (default: 500)
        overlap_percentage: Chunk'lar arası overlap yüzdesi (default: 0.1)
        output_filename: Çıkış CSV dosya adı
    
    Returns:
        pandas.DataFrame: Chunk'ları içeren DataFrame
    """
    
    encoding = tiktoken.get_encoding("cl100k_base")

    print("Corpus tokenlaştırılıyor...")
    print("type of ")
    tokens = encoding.encode(corpus_text)
    total_tokens = len(tokens)
    print(f"Toplam token sayısı: {total_tokens}")
    
    overlap_size = int(chunk_size * overlap_percentage)
    step_size = chunk_size - overlap_size
    
    print(f"Chunk boyutu: {chunk_size} token")
    print(f"Overlap: %{overlap_percentage*100} ({overlap_size} token)")
    print(f"Step boyutu: {step_size} token")
    
    chunks = []
    chunk_tokens_list = []
    
    for i in range(0, total_tokens, step_size):
        chunk_tokens = tokens[i:i + chunk_size]
        
        if len(chunk_tokens) < chunk_size * 0.5 and i > 0:
            break
        
        chunk_tokens_list.append(chunk_tokens)
        
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        print(f"Chunk {len(chunks)}: {len(chunk_tokens)} token")
        
        if i + chunk_size >= total_tokens:
            break
    
    print(f"Toplam {len(chunks)} chunk oluşturuldu")

    df = pd.DataFrame({
        'chunk_id': range(len(chunks)),
        'text': chunks,
        'token_count': [len(chunk_tokens) for chunk_tokens in chunk_tokens_list],
        'start_token_index': [i * step_size for i in range(len(chunks))],
        'end_token_index': [min(i * step_size + chunk_size, total_tokens) for i in range(len(chunks))]
    })

    df.set_index('chunk_id', inplace=True)

    df.to_csv(output_filename, encoding='utf-8', index=True)
    print(f"Chunks {output_filename} dosyasına kaydedildi")

    print("\n=== ÖZET ===")
    print(f"Toplam chunk sayısı: {len(df)}")
    print(f"Ortalama chunk boyutu: {df['token_count'].mean():.1f} token")
    print(f"Min chunk boyutu: {df['token_count'].min()} token")
    print(f"Max chunk boyutu: {df['token_count'].max()} token")
    

class FAISSIndexer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        embedding modeli ile FAISS indexer
        
        Args:
            model_name: embedding model adı
        """
        logger.info(f"Model yükleniyor: {model_name}")
        self.model = AutoModel.from_pretrained(model_name,local_files_only=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True)
        
        # FAISS index'leri
        self.tr_index = None
        self.en_index = None
        
        # Metadata'ları saklamak için
        self.tr_metadata = []
        self.en_metadata = []
    
    def embed_texts(self, texts: List[str], batch_size: int = 2) -> np.ndarray:
        """
        Metinleri embed eder
        
        Args:
            texts: Embed edilecek metinler
            batch_size: Batch boyutu
            
        Returns:
            numpy.ndarray: Embeddings matrisi
        """
        logger.info(f"{len(texts)} metin embed ediliyor...")

        encoded_input = self.tokenizer(texts,padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**encoded_input)
        
        return F.normalize(mean_pooling(outputs,encoded_input["attention_mask"]),p=2,dim=1)
    
    def create_faiss_index(self, embeddings: np.ndarray, use_gpu: bool = False) -> faiss.Index:
        """
        FAISS index oluşturur
        
        Args:
            embeddings: Embedding matrisi
            use_gpu: GPU kullanılacak mı
            
        Returns:
            faiss.Index: FAISS index
        """
        n_vectors, dim = embeddings.shape
        logger.info(f"FAISS index oluşturuluyor: {n_vectors} vector, {dim} dimension")

        index = faiss.IndexFlatIP(dim)
        
        if use_gpu and faiss.get_num_gpus() > 0:
            logger.info("GPU kullanılıyor")
            gpu_resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
        
        return index
    
    def process_corpus(self, 
                      csv_file: str, 
                      language: str, 
                      batch_size: int = 32,
                      use_gpu: bool = False) -> Tuple[faiss.Index, List[Dict]]:
        """
        CSV dosyasındaki corpus'u işler ve FAISS index oluşturur
        
        Args:
            csv_file: CSV dosya yolu
            language: 'tr' veya 'en'
            batch_size: Embedding batch boyutu
            use_gpu: GPU kullanılacak mı
            
        Returns:
            Tuple[faiss.Index, List[Dict]]: Index ve metadata
        """
        logger.info(f"{language.upper()} corpus işleniyor: {csv_file}")
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_file}")
        
        df = pd.read_csv(csv_file, index_col=0)
        logger.info(f"CSV yüklendi: {len(df)} chunk")
        
        embeddings = self.embed_texts(df['text'].tolist(), batch_size=batch_size)
        
        index = self.create_faiss_index(embeddings, use_gpu=use_gpu)
        
        logger.info("Embeddings index'e ekleniyor...")
        index.add(embeddings)
        
        metadata = []
        for idx, row in df.iterrows():
            metadata.append({
                'chunk_id': idx,
                'text': row['text'],
                'token_count': row.get('token_count', 0),
                'start_token_index': row.get('start_token_index', 0),
                'end_token_index': row.get('end_token_index', 0),
                'language': language
            })
        
        logger.info(f"{language.upper()} index oluşturuldu: {index.ntotal} vectors")
        return index, metadata
    
    def save_index(self, 
                   index: faiss.Index, 
                   metadata: List[Dict], 
                   language: str, 
                   base_path: Path):
        """
        Index ve metadata'yı kaydeder
        
        Args:
            index: FAISS index
            metadata: Metadata listesi
            language: Dil kodu
            base_path: Kayıt klasörü
        """
        os.makedirs(base_path, exist_ok=True)
        
        if hasattr(index, 'index'):
            index = faiss.index_gpu_to_cpu(index)

        index_path = os.path.join(base_path, f"{language}_index.faiss")
        faiss.write_index(index, str(index_path))
        logger.info(f"Index kaydedildi: {index_path}")
        
        metadata_path = os.path.join(base_path, f"{language}_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Metadata kaydedildi: {metadata_path}")
    
    def load_index(self, 
                   language: str, 
                   base_path: str = "./faiss_indexes",
                   use_gpu: bool = False) -> Tuple[faiss.Index, List[Dict]]:
        """
        Kaydedilmiş index ve metadata'yı yükler
        
        Args:
            language: Dil kodu
            base_path: Index klasörü
            use_gpu: GPU kullanılacak mı
            
        Returns:
            Tuple[faiss.Index, List[Dict]]: Index ve metadata
        """
        index_path = os.path.join(base_path, f"{language}_index.faiss")
        index = faiss.read_index(index_path)

        if use_gpu and faiss.get_num_gpus() > 0:
            gpu_resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)

        metadata_path = os.path.join(base_path, f"{language}_metadata.pkl")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        logger.info(f"{language.upper()} index yüklendi: {index.ntotal} vectors")
        return index, metadata
    
    def search_similar(self, 
                      query: str, 
                      language: str,
                      k: int = 5,
                      index: faiss.Index = None,
                      metadata: List[Dict] = None) -> List[Dict]:
        """
        Benzer metinleri arar
        
        Args:
            query: Arama sorgusu
            language: Hedef dil
            k: Döndürülecek sonuç sayısı
            index: FAISS index (None ise yükle)
            metadata: Metadata (None ise yükle)
            
        Returns:
            List[Dict]: Arama sonuçları
        """
        if index is None or metadata is None:
            index, metadata = self.load_index(language)

        query_embedding = self.embed_texts([query])[0:1]

        scores, indices = index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  
                result = metadata[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results

def create_faiss_indexes(tr_csv: str, 
                        en_csv: str, 
                        save_path: str,
                        batch_size: int = 32,
                        use_gpu: bool = False,
                        model_path:str = "all-MiniLM-L6-v2"):
    """
    TR ve EN corpus'lar için FAISS index'leri oluşturur
    
    Args:
        tr_csv: Türkçe corpus CSV dosyası
        en_csv: İngilizce corpus CSV dosyası  
        batch_size: Embedding batch boyutu
        use_gpu: GPU kullanılacak mı
        save_path: Index kayıt yolu
    """
    
    indexer = FAISSIndexer(model_name=model_path)
    
    if os.path.exists(tr_csv):
        logger.info("=== TÜRKÇE CORPUS İŞLENİYOR ===")
        tr_index, tr_metadata = indexer.process_corpus(
            tr_csv, 'tr', batch_size=batch_size, use_gpu=use_gpu
        )
        indexer.save_index(tr_index, tr_metadata, 'tr', save_path)
    else:
        logger.warning(f"TR CSV dosyası bulunamadı: {tr_csv}")

    if os.path.exists(en_csv):
        logger.info("=== İNGİLİZCE CORPUS İŞLENİYOR ===")
        en_index, en_metadata = indexer.process_corpus(
            en_csv, 'en', batch_size=batch_size, use_gpu=use_gpu
        )
        indexer.save_index(en_index, en_metadata, 'en', save_path)
    else:
        logger.warning(f"EN CSV dosyası bulunamadı: {en_csv}")
    
    logger.info("=== TÜM İŞLEMLER TAMAMLANDI ===")
    return indexer

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    FAISS_DIR = os.path.join(BASE_DIR, "faiss_indexes")
    CORPUS_DIR = os.path.join(BASE_DIR, "corpuses")
    MODELS_DIR = os.path.join(BASE_DIR,"models")
    MODEL_DIR = os.path.join(MODELS_DIR,"all-MiniLM-L6-v2")

    corpus_en = open(os.path.join(CORPUS_DIR,"corpus_en.txt"),"r",encoding="utf-8").read().replace("\n"," ")
    corpus_tr = open(os.path.join(CORPUS_DIR,"corpus_tr.txt"),"r",encoding="utf-8").read().replace("\n"," ")

    tokenize_and_chunk_corpus(corpus_text=corpus_en,output_filename= os.path.join(CORPUS_DIR,"chunked_corpus_en.csv"),chunk_size=300,overlap_percentage=0.2)
    tokenize_and_chunk_corpus(corpus_text=corpus_tr,output_filename= os.path.join(CORPUS_DIR,"chunked_corpus_tr.csv"),chunk_size=300,overlap_percentage=0.2)

    indexer = create_faiss_indexes(
        tr_csv=r"corpuses\chunked_corpus_tr.csv",
        en_csv=r"corpuses\chunked_corpus_en.csv", 
        batch_size=1,  
        use_gpu=True,
        save_path="./faiss_indexes",
        model_path=MODEL_DIR
    )
    
    try:
        tr_results = indexer.search_similar(
            query="What is the distance of the moon from the earth?",
            language="en", 
            k=3
        )
        
        print("\nTR Arama Sonuçları:")
        for i, result in enumerate(tr_results):
            print(f"{i+1}. Score: {result['similarity_score']:.3f}")
            print(f"   Text: {result['text'][:100]}...")
            print()
            
    except Exception as e:
        logger.error(f"Arama hatası: {e}")