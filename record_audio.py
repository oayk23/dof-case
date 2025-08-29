import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from argparse import ArgumentParser
import os

def record(output_file):
    fs = 44100
    channels = 1
    recording = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        recording.append(indata.copy())

    print("Enter’a basınca kayıt başlıyor, tekrar Enter’a basınca duruyor...")

    input("Başlamak için Enter >> ")
    print("Kayıt başladı...")

    with sd.InputStream(samplerate=fs, channels=channels, callback=callback):
        input("Bitirmek için Enter >> ")

    print("Kayıt durdu.")

    audio = np.concatenate(recording, axis=0)
    if not output_file.endswith(".wav"):
        output_file += ".wav"
    write(output_file, fs, (audio * 32767).astype(np.int16))
    print(f"Ses kaydedildi:{output_file} ")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_filename",required=True,help="Specify path of recording.")

    args = parser.parse_args()

    if not os.path.exists("audios"):
        os.makedirs("audios",exist_ok=True)
    
    record(os.path.join("audios",args.output_filename))