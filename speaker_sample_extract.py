from utils import l2_normalize, read_config, str2bool, dir_exist_check, file_type_tansforming, file_exist_check
from customized_diarization import diarization
from speechbrain.inference.speaker import EncoderClassifier
from pathlib import Path
import os 
import shutil
import librosa
import soundfile as sf
import torch
import numpy as np

class sample_extrator(diarization):
    def __init__(self, name:str, source:str , start:float = None, stop:float = None):
        super().__init__(source)
        self.SPEAKER = name
        self.INPUT_PATH = source
        self.TEMP_FILE_PATH = self.TEMP_FILE_DIR / f"{self.FILENAME}.wav"
        self.OUTPUT_DIR = self.ROOT_PATH / "./speaker_sample"
        self.EMBEDDING_OUTPUT_PATH = self.OUTPUT_DIR / f"{self.SPEAKER}_sample.npy"
        self.AUDIO_OUTPUT_PATH = self.OUTPUT_DIR / f"{self.SPEAKER}_sample.wav"
        self.START = start
        self.STOP = stop

    def processing_flow(self):
        config = read_config(self.CONFIG_PATH)

        # output directory check
        dir_exist_check(self.OUTPUT_DIR)
        dir_exist_check(self.TEMP_FILE_DIR)

        if file_exist_check(self.EMBEDDING_OUTPUT_PATH, self.print_messege):
            print(f"Speaker Embedding {os.path.basename(self.EMBEDDING_OUTPUT_PATH)} has already existed.")
            return True
        
        # file type transforming 
        file_type_tansforming(self.INPUT_PATH, self.TEMP_FILE_PATH, self.audio_to_wav)
        
        # preprocessing 
        if (self.START is not None) and (self.STOP is not None):
            self.audio_trimmer()

        self.preprocessing(self.AUDIO_OUTPUT_PATH, self.AUDIO_OUTPUT_PATH, if_vad=str2bool(config["preprocessing"]["vad"]), 
                           if_vad_keep_length=str2bool(config["preprocessing"]["vad_keep_length"]), 
                           if_preemphasis=str2bool(config["preprocessing"]["pre_emphasis"]))

        # speaker embedding
        if config["init"]["embed"].lower() == "ecapa":
            wav, _ = librosa.load(self.AUDIO_OUTPUT_PATH, sr=self.SAMPLE_RATE)
            inputs = torch.tensor(wav).to(self.DEVICE)
            model = EncoderClassifier.from_hparams(source=self.ENCODERCLASSIFIER_PATH)
            model = model.to(self.DEVICE)
            embedding = self.ECAPA_embedding(inputs, model)
            embedding = embedding.reshape(-1, embedding.shape[-1]).cpu().numpy()
            print("embedding array shape: {}".format(embedding.shape))
        else:
            raise ValueError("The embedding method does not exist.")
        
        # normalization
        if str2bool(config["init"]["normalizartion"]):
            embedding = l2_normalize(embedding)
            print("Normalization: True")
        else:
            print("Normalization: False")

        np.save(self.EMBEDDING_OUTPUT_PATH, embedding)

    def audio_trimmer(self):
        wav, _ = librosa.load(self.INPUT_PATH, sr=self.SAMPLE_RATE)
        sf.write(self.AUDIO_OUTPUT_PATH, wav[int(self.START*self.SAMPLE_RATE): int(self.STOP*self.SAMPLE_RATE)], self.SAMPLE_RATE)
        return wav[int(self.START*self.SAMPLE_RATE): int(self.STOP*self.SAMPLE_RATE)]
    
    def print_messege(*args):
        print(f"Speaker Embedding Path: {args[-1]}")
    
if __name__ == "__main__":
    # x = sample_extrator("Jeremy", "./audio_data/Emotions - 30 Sec.mp3", 0, 10 )
    x = sample_extrator("Jeremy", "speaker_sample\Jeremy_sample.wav")
    x.processing_flow()