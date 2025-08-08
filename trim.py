import librosa
import soundfile as sf
import json
import re
import os
from pathlib import Path
from utils import video_to_audio, dir_exist_check, file_type_tansforming, valid_input_type_check
from argparse import ArgumentParser

def media_trim(input_path: str|Path, start: int, stop: int):
    sample_rate = 16000
    temp_dir = Path("./temp")
    audio_dir = Path("./audio_data")

    dir_exist_check(temp_dir)
    dir_exist_check(audio_dir)
    filename, extension = ".".join(os.path.basename(input_path).split(".")[:-1]), os.path.basename(input_path).split(".")[-1]

    output_path_for_file_type_convert = temp_dir / f'{filename}.wav'
    output_path_for_media_trim = audio_dir / f"{filename}_trim_{start}_{stop}.wav"

    file_type = valid_input_type_check(input_path)
    if file_type == "video":
        video_to_audio(input_path, output_path_for_file_type_convert)
        wav, _ = librosa.load(output_path_for_file_type_convert, sr=sample_rate)
        sf.write(output_path_for_media_trim, wav[int(start*sample_rate): int(stop*sample_rate)], sample_rate)
        os.remove(output_path_for_file_type_convert)
    elif file_type == "audio":
        wav, _ = librosa.load(input_path, sr=sample_rate)
        sf.write(output_path_for_media_trim, wav[int(start*sample_rate): int(stop*sample_rate)], sample_rate)
    else:
        raise TypeError("Unsupport file type") 
    
def transcription_trim(input_path: str|Path, start: int, stop: int):
    
    trans_dir = Path("transcription_data")
    dir_exist_check(trans_dir)
    filename, extension = ".".join(os.path.basename(input_path).split(".")[:-1]), os.path.basename(input_path).split(".")[-1]
    output_path_for_trans_trim = trans_dir / f"{filename}_trim_{start}_{stop}.txt"

    transcription_lst = []
    with open(input_path, 'r', encoding='utf-8') as f:
        regex = r"\[(\d+\.\d{2})s -> (\d+\.\d{2})s\]"
        for row in f.readlines():
            t_start, t_end = re.match(regex, row).group(1,2)
            if start <= float(t_start):
                update_row = re.sub(regex, f"[{(float(t_start)-start):.2f}s -> {(float(t_end)-start):.2f}s]", row)
                transcription_lst.append(update_row)
                print(update_row)
            if stop < float(t_end):
                break

    with open(output_path_for_trans_trim, "w", encoding='utf-8') as o:
        o.writelines(transcription_lst)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-mp", "--media_path", help="input media path", dest="media_path", type=str, required=False, default=None)
    parser.add_argument("-tp", "--transcription_path", help="transcription path", dest="transcription_path", type=str, required=False, default=None)
    parser.add_argument("-str", "--start", help="The start time in seconds", dest="start", type=int, required=True)
    parser.add_argument("-stp", "--stop", help="The stop time in seconds", dest="stop", type=int, required=True)
    args = parser.parse_args()

    # input_path = "./video_data/20250805102550audio.mp4"
    if args.media_path:
        print(f"trimming {args.media_path} from {args.start}s to {args.stop}s...")
        media_trim(args.media_path, args.start, args.stop)
    if args.transcription_path:
        print(f"trimming {args.transcription_path} from {args.start}s to {args.stop}s...")
        transcription_trim(args.transcription_path, args.start, args.stop)

    
#  python trim.py -mp "./video_data/20250805102550audio.mp4" -tp "video_data\\20250805102550audio_TIME.txt"