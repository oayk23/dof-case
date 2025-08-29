# DOF Robotics AI Engineer Case Study
---
Bu Case Study'de RAG Tabanlı sesli sohbet sistemi geliştirilmesi amaçlanmıştır. 
Kullanılan Bileşenler:
- STT(Speech-to-text) için Faster-Whisper-Base modeli
- RAG Retrieval olarak sentence-transformers/all-MiniLM-L6-v2 modeli
- LLM olarak Qwen3/1.7B modeli
- TTS(Text-to-Speech) için coqui/xtts-v2 modeli
Kullanılmıştır.
---
## Kurulum

Reponun başarı ile kurulabilmesi için Python'un 3.11.0 versiyonu bilgisayarda bulunmalıdır. Uygulamanın test edildiği sitemin özellikleri: 64GB RAM, 8GB RTX 2070 GPU dur. Önerilen sistem özellikleri minimum 32GB RAM ve 8GB GPU'dur.

Kurulum Aşamaları:

1. Sisteminize Ollama uygulamasını kurun. [Link](https://ollama.com/download/windows)
2. ```git clone https://github.com/oayk23/dof-case``` komutu ile masaüstünde bir terminal açarak bu repo'yu bilgisayarınıza indirin.
3. setup.bat dosyasına çift tıkladığınızda kurulum başlayacaktır. Kurulum başlarken internet bağlantısı gereklidir.

Bu aşamalardan sonra sistem çalışmaya hazır hale gelecektir.

---
## Uygulamayı çalıştırmak ve örnek sonuç almak

Uygulama kurulduktan sonra uygulamanın bulunduğu dizinde iki tane terminal açılması gerekmektedir. Bir terminal kullanıcıdan ses kaydetmesini sağlarken diğer terminal kaydedilen sesin uygulama tarafından işlenip ses oluşturmasını sağlayacaktır.
1. İlk terminali açın ve:
```
.\dof\Scripts\activate
python .\record_audio.py --output_filename example
```
komutlarını çalıştırın.

Burada --output_filename bayrağı sizin oluşturduğunuz sesin; audios/ klasöründe hangi isimle kaydedileceğini belirler. Dolayısıyla farklı sesler için farklı isimlendirmeler kullanılması önerilmektedir aksi halde önceki sesin üzerine kaydedilecektir.

2. İkinci terminalde:
```
.\dof\Scripts\activate
python .\main.py
```

Komutlarını çalıştırın. Bu terminal size uygulamayı açacaktır. Uygulamanın işleyiş mantığı şu şekildedir:

- Öncelikle size hangi sesi işlemek istediğiniz sorulacaktır. Burada ilk terminalde kaydettiğiniz sesin ismini girmeniz yeterlidir. Program bunu çalıştırıp Pipeline'a sesi gönderip aynı ismin sonuna "_response" ekleyerek sesi kaydedecektir.
- Girdiğiniz ismin (örneğin regolit) sonuna "_response" eklenerek (örneğin regolit_response) kaydedilen sesi dinleyerek output'u değerlendirebilirsiniz.

---

## Mimari

Daha öncede belirtildiği üzere yukarıdaki modeller kullanılmıştır.
Bunların haricinde vector database olarak FAISS kullanılmıştır.
RAG için elde edilen corpus manuel olarak txt dosyasına kaydedilmiştir ve repoda bulunmaktadır.
Veri akışı şu şekildedir:
1. Input verisi denoise edilir, yani ortam gürültüsü noisereduce kütüphanesi kullanılarak azaltılır ve input+"_denoised" olarak aynı dizine kaydedilir.
2. Denoised Input Speech-to-text pipeline'ına verilip bu pipeline'dan elde edilen text ve language (dil) bilgileri alınır.
3. Elde edilen text ve language değişkenleri FAISS Tabanlı vektör araması için kullanılmaktadır. Text RAG sisteminden en yakın sonuç getirilmesi amacıyla bir embedding modeline verilir ve hangi language ise o dilin bulunduğu vektör database'den en yakın sonuç getirilir.
4. Getirilen sonuçlar doküman olarak LLM Pipeline'ına verilir ve Ollama-python apisi kullanılarak kullanıcının girdiği dilde bir yanıt oluşturulur.
5. LLM tarafından oluşturulan yanıt Text-to-Speech pipeline'ına verilir. Bu pipeline da erkek veya kadın sesi olmasına göre (koddaki varsayılan değer kadındır.) metni cümle cümle sese dönüştürüp en sonunda bu sesleri birleştirerek tek bir nihai ses oluşturmaktadır.

---
## Kurulumda yapılan işler

Kurulum sırasında bağımlılıklar vs. kurulur ve en sonunda indexer.py modülü __main__ olarak çalıştırılıp corpusu 300 tokenlık chunklara böler. Bu işlem sırasında tiktoken kütüphanesi kullanılacağı için bu işlem internet gerektirmektedir. corpus 300 tokenlık chunklara bölünüp %20 overlap eklendiği zaman elde edilen textler ise bir csv dosyasına aktarılır.
Daha sonrasında bu csv dosyasındaki her bir text RAG embedder kullanılarak embed edilip dillerine göre FAISS vector databaselerine eklenir.

## Guardrail mantığı ve eşik değer

Uygulama testlerinde guardrail konulduğu zaman, yani belli bir seviyenin altında bir similarity varsa bunu doküman olarak LLM'e verme dendiği zaman sonuçlar çok kötü olmaktadır. Dolayısıyla guardrail mantığı yerine __prompt engineering__ ile LLM'e system promptunda "Ay(Moon)" dışındaki sorulara cevap verme şeklinde bir prompt girilmiştir.
Bu mantık guardrailden daha iyi sonuç vermektedir.

## Mevcut Kısıtlamalar ve gelecekteki iyileştirmeler

Sistemde bulunan en büyük kısıt kaynaklardır. 

- Whisper'ın daha iyi modeli (medium, large vb.) kullanılarak Speech-to-text te daha iyi sonuçlar alınabilir. Örneğin kendi sesimle regolit dediğim zaman, STT modülü bunu "regulit" diye anlayabiliyor ve bu da RAG sisteminin ve LLM'in performansını düşürüyor.
Bu sorunun çözümü olarak ileride base gibi düşük parametreli modelleri domain specific fine-tuning gerçekleştirilebilir. Örneğin ay için Regolith, Lunar, Apollo,Mare gibi seslerin hem türkçe hem de ingilizce datalarıyla bir fine tuning yapılabilir. Bunun sistemin performansını oldukça iyileştireceğini düşünüyorum.
- RAG sistemi için daha geniş çaplı bir corpus hazırlanması sonucunda daha çeşitli bir corpus elde edilebilir. Bu da daha relevant sonuçlar getirebilir.
Bunun haricinde daha iyi bir embedding modeli kullanılabilir, örneğin qwen3-embedding-0.6B gibi fakat bu modeller de en nihayetinde yüksek boyutlu modeller ve bu noktada tekrardan kaynak kısıtı yaşanabilir.
- LLM konusunda mevcut kısıtlamalar: Modelin instruction ve context following yetenekleri oldukça kısıtlı. Mesela 1 yerine 2 tane en ilişkili doküman verildiğinde model uzun context'ten dolayı system prompt'unu takip edemeyebiliyor. Genel olarak karşılaşılan zorluk prompt following yeteneğidir.
Bunun için önerilen çözüm yöntemi ise ortalama bir dataset(300-500 sample'lık) ile lora fine-tuning fazlasıyla etkili olur diye düşünüyorum. Özellikle domain specific productionlarda bunun model performansını fazlasıyla iyileştireceği kanısındayım.
- TTS pipeline'ında karşılaşılan bir sorun olarak tek bir text'i işleyememesi idi. Fakat bunu da text'i cümlelere bölüp her bir cümleyi ayrı bir speech olarak çıktı alıp en sonunda bu çıktıları birleştirmek ile çözüm üretilmiştir.
