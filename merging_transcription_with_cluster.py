import re
import json
from argparse import ArgumentParser
from utils import dir_exist_check
from pathlib import Path
import os

class merging():
    def __init__(self, transcript_path, cluster_path = None):
        self.ROOT_PATH = Path(".")
        self.TRANSCRIPT_PATH = transcript_path
        self.CLUSTER_PATH = cluster_path
        self.FILENAME, self.EXTENSION = os.path.basename(self.TRANSCRIPT_PATH).split(".")
        self.OUTPUT_DIRECTORY = self.ROOT_PATH / "./output"
        self.OUTPUT_PATH = self.OUTPUT_DIRECTORY / f"{self.FILENAME}.{self.EXTENSION}"
    
    def merge(self):
        """
        This function concatenates the transcription with the corresponding speakers/clusters row by row.
        """
        with open(self.CLUSTER_PATH, 'r', encoding='utf-8') as c:
            cluster = json.loads(c.read())
            # print(cluster)

        annoted_transcription_lst = []
        with open(self.TRANSCRIPT_PATH, 'r', encoding='utf-8') as f:
            regex = r"\[(\d+\.\d{2})s -> (\d+\.\d{2})s\]"
            for row in f.readlines():
                start, end = re.match(regex, row).group(1,2)

                speaker = []
                for c in cluster["diarization"]:
                    if c["start"] <= float(start) and c["end"] >= float(end):
                        speaker.append(str(c['cluster']))

                time_format = re.match(regex, row).group(0)
                if speaker:
                    speaker = list(set(speaker))
                    # 可想想能不能更好
                    if "-1" in speaker and len(speaker) > 1:
                        speaker.remove("-1")

                    annoted_transcription_lst.append(re.sub(regex, f"{time_format} [{', '.join(speaker)}]", row))
                    # print(re.sub(regex, f"{time_format} [{c['cluster']}]", row))
                
                else:
                    annoted_transcription_lst.append(re.sub(regex, f"{time_format} [-1]", row))

        annoted_transcription_lst = list(map(lambda x: f"{x}", annoted_transcription_lst))
        dir_exist_check(self.OUTPUT_DIRECTORY)

        with open(self.OUTPUT_PATH, "w", encoding='utf-8') as o:
            o.writelines(annoted_transcription_lst)

def call(transcription_path:str|Path, cluster_path: str|Path):
    x = merging(transcription_path, cluster_path)
    x.merge()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-tp", "--transcription_path", help="transcription path", dest="transcription_path", type=str, required=True)
    parser.add_argument("-cp", "--cluster_path", help="clutser path", type=str, dest="cluster_path", required=True)
    args = parser.parse_args()

    x = merging(args.transcription_path, args.cluster_path)
    x.merge()