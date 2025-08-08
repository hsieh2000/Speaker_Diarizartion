import customized_diarization
import merging_transcription_with_cluster
from speaker_sample_extract import sample_extrator
import json
import os
from pathlib import Path
import shutil
from argparse import ArgumentParser

def filter_function(speaker_anchor:str|Path ="speaker_anchor.json", speaker:list[str] = [], source:list[str] = []) -> list[dict]:
    """
    This function filters the speakers or source files for which embeddings will be computed for speaker identification.
    If `speakers` are provided, the function computes embeddings for the speakers whose names match the given values.
    If `source` is provided, the function computes embeddings for the speakers whose source files match the given values.

    Inputs:
        @param speaker_anchor:  The path to the json file of speaker anchor infomation.
        @param speaker:         A list of speaker names to filter.
        @param source:          A list of source paths to filter.
    Return:
        List[Dict]: A list of elements from `speaker_anchor` for which embeddings will be computed.
    """
    with open(speaker_anchor, "r", encoding="utf-8") as f:
        sample_dict = json.loads(f.read())

    if source:
        source = list(map(lambda path:  os.path.abspath(path), source))

    result = list(filter(lambda x: (x["name"] in speaker or speaker == []) and \
                (os.path.abspath(x["source"]) in source or source == []), sample_dict["speaker"]))
    
    # [i.update({"source": os.path.abspath(i["source"])}) for i in result]
    
    result = sorted(result, key=lambda x: x['source'])
    print(result)
    return result

def call(media_file: str|Path, transcription_file: str|Path = None, speaker:list[str] = [], source:list[str] = [], speaker_anchor: str|Path = "speaker_anchor.json"):
    result = filter_function(speaker_anchor=speaker_anchor, speaker=speaker, source=source) 

    for anchor_info in result:
        x = sample_extrator(**anchor_info)
        x.processing_flow()
        anchor_info.update({
            "output": {
                "embedding":x.EMBEDDING_OUTPUT_PATH,
                "audio": x.AUDIO_OUTPUT_PATH
            }
        })
    
    sample_dict = {"speaker": result}
    cluster_file = customized_diarization.call(media_file, sample_dict)
    if transcription_file:
        print("mapping clusters to transcription...")
        merging_transcription_with_cluster.call(transcription_file, cluster_file)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-mp", "--media_path", help="input media path", dest="media_path", type=str, required=True)
    parser.add_argument("-tp", "--transcription_path", help="transcription path", dest="transcription_path", type=str, required=False)
    parser.add_argument("-spk", "--speaker", help="The string assigned speakers join with `|`, e.g. `Jeremy|Barry`", dest="speaker", type=str, required=False)
    parser.add_argument("-src", "--source", help="The string assigned source join with `|`, e.g. `./file1.mp4|./file2.mp4`", dest="source", type=str, required=False)
    args = parser.parse_args()

    mf = args.media_path
    tf = args.transcription_path
    spk = args.speaker.split("|") if args.speaker is not None else []
    src = args.source.split("|") if args.source is not None else []

    call(media_file=mf, transcription_file=tf, speaker=spk, source=src)

# python run.py -mp "audio_data/20250805102550audio_trim_450_513.wav" -tp "video_data\\20250805102550audio_TIME.txt" -spk "陳俊 宏|李副總|蘇茂凱"
