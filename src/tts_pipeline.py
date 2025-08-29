from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig
import os
import soundfile as sf
import re
import numpy as np

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

class TTSPipeline:
    def __init__(self,model_path,speaker_path):
        self.config = XttsConfig()
        self.config.load_json(os.path.join(model_path,"config.json"))
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=model_path, eval=True)
        self.model.cuda()
        self.speaker_path = speaker_path
    
    def split_text(self,text, max_sentences=1):
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunk = " ".join(sentences[i:i+max_sentences])
            chunks.append(chunk)
        return chunks

    def __call__(self,text,language,input_audio_path):
        chunks = self.split_text(text)
        audio_arrays = []
        for idx, chunk in enumerate(chunks):

            output = self.model.synthesize(chunk,self.config,speaker_wav=self.speaker_path,language=language)
            audio = output["wav"]
            audio_arrays.append(audio)

        final_audio = np.concatenate(audio_arrays)

        output_path = input_audio_path.split(".")[0] + "_response.wav"

        sf.write(output_path,final_audio,samplerate=24000)
