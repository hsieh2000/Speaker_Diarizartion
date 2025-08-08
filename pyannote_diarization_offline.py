import os
import json
from pathlib import Path
import shutil
from pyannote.audio import Pipeline
from IPython.display import Audio
from argparse import ArgumentParser
import time
from utils import vad, pre_emphasis, valid_input_type_check, video_to_audio, \
    timestamp_recover, str2bool, dir_exist_check, output, file_exist_check, read_config

import warnings
warnings.filterwarnings('ignore')


class diarization():
    def __init__(self, input_path: str):
        self.ROOT_PATH = Path(".")
        # Because of the restrictions in the `huggingface_hub` package on the format of the repo ID, the model path must follow the format of either "repo_name" or "namespace/repo_name".  
        # More detailed naming rules can be found the `validate_repo_id` function of `Lib/site-packages/huggingface_hub/utils/_validators.py`.
        self.PYANNOTE_MODEL_CONFIG_PATH = self.ROOT_PATH / "pyannote_models/pyannote_diarization_config.yaml" 
        self.INPUT_PATH =  input_path
        self.FILENAME, self.FILE_EXTENSION = ".".join(os.path.basename(self.INPUT_PATH).split(".")[:-1]), os.path.basename(self.INPUT_PATH).split('.')[-1]
        self.TEMP_FILE_DIR = "./temp"
        self.TEMP_FILE_PATH = self.ROOT_PATH / self.TEMP_FILE_DIR / f"{self.FILENAME }.wav"
        self.VAD_MODULE_PATH = self.ROOT_PATH / "silero-vad"  
        self.OUTPUT_DIR = self.ROOT_PATH / "json_output"
        self.module = "pyannote_diarization"

    def diarization(self, input_file_path: str):
        """
        @param input_file_path: Path to the input audio file.
        """
        pipeline = self._load_pipeline_from_pretrained(self.PYANNOTE_MODEL_CONFIG_PATH)
        diarization = pipeline(input_file_path)
        output = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"start={turn.start:.2f}s stop={turn.end:.2f}s speaker_{speaker}")
            output.append(
                {
                    "start": round(turn.start, 2), 
                    "end": round(turn.end, 2),
                    "cluster": int(speaker.split('_')[1])
                }
            )        

        return output          

    def _load_pipeline_from_pretrained(self, path_to_config: str | Path) -> Pipeline:
        """
         @param path_to_config: Path to the configuration file.
        """
        path_to_config = Path(path_to_config)

        print(f"Loading pyannote pipeline from {path_to_config}...")
        # the paths in the config are relative to the current working directory
        # so we need to change the working directory to the model path
        # and then change it back

        cwd = Path.cwd().resolve()  # store current working directory

        # first .parent is the folder of the config, second .parent is the folder containing the 'models' folder
        cd_to = path_to_config.parent.parent.resolve()

        print(f"Changing working directory to {cd_to}")
        os.chdir(cd_to)

        pipeline = Pipeline.from_pretrained(path_to_config)

        print(f"Changing working directory back to {cwd}")
        os.chdir(cwd)

        return pipeline
    
def call(input_path):
    # initializing instance and loading configuration file
    x = diarization(input_path)
    x.CONFIG_PATH = x.ROOT_PATH / "./pyannote_diarization_config.json"
    config = read_config(x.CONFIG_PATH)

    # output directory check
    dir_exist_check(x.OUTPUT_DIR)
    dir_exist_check(x.TEMP_FILE_DIR)

    # file type transforming 
    file_type = valid_input_type_check(x.INPUT_PATH)
    if file_type == "video":
        video_to_audio(x.INPUT_PATH, x.TEMP_FILE_PATH)
    elif file_type == "audio":
        file_exist_check(x.TEMP_FILE_PATH, shutil.copyfile, x.INPUT_PATH)
    elif file_type == False:
        raise TypeError("Input must be the str of a file path.") 
    elif file_type is None:
        raise ValueError("Not a video or audio file.")
    
    # preprocessing 
    if str2bool(config["preprocessing"]["vad"]):
        time_mapping_lst = vad(x.TEMP_FILE_PATH, x.TEMP_FILE_PATH, x.VAD_MODULE_PATH, keep_length=str2bool(config["preprocessing"]["vad_keep_length"]))
    if str2bool(config["preprocessing"]["pre_emphasis"]):
        pre_emphasis(input_path = x.TEMP_FILE_PATH, output_path = x.TEMP_FILE_PATH)

    # diarization
    result = x.diarization(x.TEMP_FILE_PATH)
    if not str2bool(config["preprocessing"]["vad_keep_length"]):
        result = timestamp_recover(time_mapping_lst, result)

    # output
    output(result, x.TEMP_FILE_PATH, x.OUTPUT_DIR, x.module)

    # clean temp folder
    for filename in os.listdir(x.TEMP_FILE_DIR):
        file_path = os.path.join(x.TEMP_FILE_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", help="input audio path", dest="path", type=str, required=True)
    args = parser.parse_args()
    call(args.path)

# Reference:
# offline pyannote speaker-diarization-3.1
# https://github.com/pyannote/pyannote-audio/blob/main/tutorials/community/offline_usage_speaker_diarization.ipynb

# silero-vad
# https://github.com/snakers4/silero-vad/blob/master/silero-vad.ipynb
