import faster_whisper
import noisereduce as nr
import librosa
import soundfile as sf
import os

class STTPipeline:
    def __init__(self,model_path:str):

        self.model = faster_whisper.WhisperModel(model_path,local_files_only=True,device="cuda",compute_type="float16")

    def noise_reduce(self,audio_path):
        y, sr = librosa.load(audio_path, sr=None)
    
        reduced_noise = nr.reduce_noise(y=y, sr=sr)
        audio_path_corrected = audio_path.split(".")[0] + "_denoised.wav"
        sf.write(audio_path_corrected, reduced_noise, sr)
        return audio_path_corrected

    def __call__(self,audio_path:str):
        denoised = self.noise_reduce(audio_path=audio_path)
        segments,info = self.model.transcribe(denoised)
        language = info.language
        text = next(iter(segments)).text
        print("text:")
        print(text)
        return text,language