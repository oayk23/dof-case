from src.llm_pipeline import LLMPipeline
from src.stt_pipeline import STTPipeline
from src.tts_pipeline import TTSPipeline
from indexer import FAISSIndexer
import os
import warnings

warnings.filterwarnings("ignore")

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

class Pipeline:
    def __init__(self,models_dir):
        self.stt_pipe = STTPipeline(os.path.join(models_dir,"faster-whisper-base"))
        print("Speech-to-text modülü yüklendi...")
        self.llm_pipe = LLMPipeline()
        print("LLM modülü yüklendi...")
        self.indexer = FAISSIndexer(os.path.join(models_dir,"all-MiniLM-L6-v2"))
        print("RAG Modülü yüklendi...")
        self.tts_pipe = TTSPipeline(os.path.join(models_dir,"XTTS-v2"),r"speakers\female.mp3")

    def __call__(self,audio_path):
        text,language = self.stt_pipe(audio_path)
        print("stt text:")
        print(text)
        if language not in ["en","tr"]:
            search_results = self.indexer.search_similar(text,"en",k=1)
        #if search_results is not None and search_results[0]["similarity_score"] < 0.3:
        #    docs = None
        else:
            search_results = self.indexer.search_similar(text,language,k=1)
        
        docs = search_results[0]["text"]
        similarity = search_results[0]["similarity_score"]
        #if similarity < 0.55:
        #    docs = "No relative document found."
        print("dokümanlar:")
        print(docs)
        llm_response = self.llm_pipe(text,docs,language)
        print("llm response:")
        print(llm_response)
        if language not in ["en","tr"]:
            language = "en"
        speech = self.tts_pipe(llm_response,language,input_audio_path=audio_path)

if __name__ == "__main__":
    pipe = Pipeline("models")

    audios_dir = "audios"
    while True:
        audio_path = input("Ses dosyası yolunu giriniz (çıkmak için 'q'): ")

        if audio_path.lower() == "q":
            print("Programdan çıkılıyor...")
            break
        if not audio_path.endswith(".wav"):
            audio_path += ".wav"
        if not os.path.exists(os.path.join("audios",audio_path)):
            print("Geçersiz path, tekrar deneyiniz.")
            continue
        
        else:
            pipe(os.path.join("audios",audio_path))
        
