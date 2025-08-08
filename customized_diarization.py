import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
# from transformers import Wav2Vec2Model, Wav2Vec2Processor
from speechbrain.inference.speaker import EncoderClassifier
import librosa
import soundfile as sf
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import os 
from pathlib import Path
import math
from argparse import ArgumentParser
from utils import vad, pre_emphasis, l2_normalize, dimension_reduction, \
    clustering, str2bool, read_config, dir_exist_check, \
    gpu_available, timestamp_recover, output, file_type_tansforming
import warnings
warnings.filterwarnings('ignore')

class diarization():
    def __init__(self, input_path: str):
        self.ROOT_PATH = Path(".")
        self.INPUT_PATH =  input_path
        self.FILENAME, self.FILE_EXTENSION = ".".join(os.path.basename(self.INPUT_PATH).split(".")[:-1]), os.path.basename(self.INPUT_PATH).split('.')[-1]
        self.TEMP_FILE_DIR = self.ROOT_PATH / "./temp"
        self.TEMP_FILE_PATH = self.TEMP_FILE_DIR / f"{self.FILENAME }.wav"
        self.VAD_MODULE_PATH = self.ROOT_PATH / "./silero-vad"  
        self.CONFIG_PATH = self.ROOT_PATH / "./customized_diarization_config.json"
        self.OUTPUT_DIR = self.ROOT_PATH / "./json_output"
        self.ENCODERCLASSIFIER_PATH = self.ROOT_PATH / "./spkrec-ecapa-voxceleb" # Don't forget to change the `pretrained_path` parameter in spkrec-ecapa-voxceleb/hyperparams.yaml as well. 
        self.DEVICE = gpu_available()
        self.SAMPLE_RATE = 16000
        self.module = "customized_diarization"
    
    def preprocessing(self, input_path, output_path, if_vad:bool = True, if_preemphasis:bool = True, if_vad_keep_length:bool = False) -> list[dict]|None:
        """
        This function organizes the preprocessing flow.
        Inputs:
            @param input_path:          str|Path,   Path to the input file.
            @param output_path:         str|Path,   Path to the output file.
            @param if_vad:              bool,       To identify whether VAD will apply to the audio file or not.
            @param if_preemphasis:      bool,       To identify whether pre-emphasisthe will apply to the audio file or not.
            @param if_vad_keep_length:  bool,       To identify whether the audio file is trimmed or not during VAD.
        Return:
            list[dict]|None, if `if_vad_keep_length` is False, the function will return a list[dict] including the original - trimmed time mapping information. Otherwise, return None.
        """
        if if_vad: 
            print(input_path)
            time_mapping_lst = vad(input_path, output_path, self.VAD_MODULE_PATH, keep_length=if_vad_keep_length)
        if if_preemphasis:
            pre_emphasis(input_path=output_path, output_path=output_path, sample_rate=self.SAMPLE_RATE)   
        
        return time_mapping_lst
    
    def audio_to_wav(self, input_path: str|Path, output_path: str|Path) -> bool:
        """
        This function convert the audio file format to WAV.
        Inputs:
            @param input_path:  str|Path, Path to the input file.
            @param output_path: str|Path, Path to the output file.
        Return:
            bool, True
        """
        wav, sr = librosa.load(input_path, sr=self.SAMPLE_RATE)
        sf.write(output_path, wav, sr)
        return True

    def ECAPA_embedding(self, _inputs, model) -> torch.tensor:
        """
        This function computes the embedding for the input audio.
        Inputs:
            @param _inputs: Tensor, the tensor of size B x T x *.
            @param model:   int, sampling rate of the input audio file.
        Return:
            torch.tensor, the embedding for the input audio.
        """
        # _inputs = torch.tensor(_inputs).to(self.DEVICE)
        embeddings = model.encode_batch(_inputs)
        embeddings = torch.squeeze(embeddings)
        return embeddings

    def wav_segmentation(self, wav_path:str, embedding_model:object, window:float = 1.6, stride:float = 0.08, \
                         batch_size:int = 1, vad_keep_length:bool = False, time_mapping_lst:list[dict] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        This function handles the workflow of audio segmentation.
        Inputs:
            @param wav_path:            str,        Path to the input audio file.
            @param embedding_model:     object,     the embedding function applied to input audio file.
            @param window:              int,        size of the window.
            @param stride:              int,        size of the stride.
            @param batch_size:          int,        size of each batch.
            @param vad_keep_length:     bool,       to identify whether the audio file is trimmed or not.
            @param time_mapping_lst:    list[dict], each containing the original and trimmed start/end timestamps corresponding to segments within the audio file.
        Return:
            tuple[np.ndarray, np.ndarray], the first output is the array of audio segments and the second output is the `start`, `stop` timestamp of the corresponding segments.
        """
        wav, sr = librosa.load(wav_path, sr=self.SAMPLE_RATE)

        if embedding_model == self.ECAPA_embedding:
            model = EncoderClassifier.from_hparams(source=self.ENCODERCLASSIFIER_PATH)
            model = model.to(self.DEVICE)
        else:
            return None

        window_size = int(window*sr)
        stride_size = int(stride * sr)
        wav_size = int(wav.shape[0])
        segs_count = math.ceil((wav_size - window_size)/stride_size) + 1
        output = []
        for batch_iter in tqdm(range(math.ceil(segs_count/batch_size))):
            inputs = pad_sequence([torch.tensor(wav[(b+batch_iter*batch_size)*stride_size : (b+batch_iter*batch_size)*stride_size+window_size]) for b in range(batch_size)], \
                                   batch_first=True, padding_value=0).to(self.DEVICE)
            batch_embedding = embedding_model(inputs, model)
            output.append(batch_embedding)

        output = torch.stack(output, dim=0).cpu().numpy()
        output = output.reshape(-1, output.shape[-1])
        
        timestamps = np.array([(x*stride_size/sr, wav_size/sr) if x == segs_count-1 else (x*stride_size/sr, (x*stride_size + window_size)/sr) for x in range(segs_count)])
        return (output, timestamps)

    def assign_clusters(self, wav_splits, clustering) -> list[dict]:
        """
        This function concatenates the timestamps of segments with the corresponding cluster.
        Inputs:
            @param wav_splits: numpy.ndarray, The second output of wav_segmentation, the start and the stop time(ms) of each segment, its length should be the same with the length of clustering.
            @param clustering: numpy.ndarray, The cluster of each segment.
        Return:
            list[dict], a list of dictionary includes the `start`, `end` and `cluster` of each segment.
        """
        return [{"start": (item[0]), "end": (item[1]), "cluster": int(c)} for item, c in zip(wav_splits, clustering)]

    def segmentation_annote_with_clusters(self, clustered_seg_lst: list[dict], min_noise_ignore:int = 5, report_seg_cnt:bool = True) -> list[dict]:
        """
        This function is used to merge adjacent segments that belong to the same cluster and remove segments labeled as noise (-1). 

        Inputs:
            @param clustered_seg_lst:   List of Dictonaries, The output of assign_clusters().
            @param min_noise_ignore:    Int, the number used to filter out items from the noise cluster. If the number of segments is lower than this value, the item will be filtered out. Default is 5.
            @param report_seg_cnt:      Bool, whether to include the count of segments in the returned result. Default is True.

        Return:
            list[dict], As the same format as the output of assign_clusters().
        """
        pre_seg = None
        start_idx = 0
        start_time = 0
        seg_lst = []

        for idx, current_seg in enumerate(clustered_seg_lst):
            if pre_seg:
                if pre_seg['cluster'] != current_seg['cluster']:
                    # print(f'start_time: {start_time}, end_time: {pre_seg["end"]}, cluster: {pre_seg["cluster"]}')
                    _dict = {
                    "start": float(start_time),
                    "end": float(pre_seg["end"]),
                    "cluster": str(pre_seg["cluster"]),
                    }
                    if report_seg_cnt:
                        _dict.update({"seg_cnt": int(idx - start_idx)})

                    seg_lst.append(_dict)
                    # record the beginning timestamp of current cluster
                    start_idx, start_time = idx, current_seg['start']
            if idx+1 == len(clustered_seg_lst):
                # print(f'start_time: {start_time}, end_time: {current_seg["end"]}, cluster: {current_seg["cluster"]}')
                _dict = {
                    "start": float(start_time),
                    "end": float(current_seg["end"]),
                    "cluster": str(current_seg["cluster"]),
                }
                if report_seg_cnt:
                    _dict.update({"seg_cnt": int(idx - start_idx)})

                seg_lst.append(_dict)
            pre_seg = current_seg
        if min_noise_ignore > 0:
            seg_lst = [i for i in filter(lambda x: (x['seg_cnt'] > min_noise_ignore) or (x['cluster'] != -1), seg_lst)]

        return seg_lst

    def speaker_recognition(self, clustered_seg_lst:list[dict], most_similar_speaker:list[str], similarity_ratio:list[np.float64], threshold: float = 0.7) -> list[dict]:
        """
        This function replaces the cluster of segment with actual speakers by computing the cosine similarity between speakers embedding and segment embedding.
        Inputs:
            @param clustered_seg_lst:       list[dict],         The output of assign_cluster().
            @param most_similar_speaker:    list[str],          The corresponding name of the speaker embeddings having the highest cosine similarity score with the audio segment.
            @param similarity_ratio:        list[np.float64],   The highest cosine similarity score between each audio segment embedding and the filtered speaker embeddings.
            @param threshold:               float,              The similarity threshold for deciding if an audio segment belongs to one of the filtered (known) speakers.
        Return:
            list[dict], The same format as the output of assign_cluster().
        """
        for i, j, k in zip(clustered_seg_lst, most_similar_speaker, similarity_ratio):
            i.update({
                "cluster": "{}".format(j) if k.astype(float) > threshold else i["cluster"]
            }) 
        return clustered_seg_lst
    
    def speaker_cluster_smoothing(self, result: list[dict]) -> list[dict]:
        """
        This function smoothes the output of assign_cluster().
        Inputs:
            @param result: list[dict], The output of assign_cluster().
        Return:
            list[dict], The same format as the output of assign_cluster().
        """
        for s in range(len(result)):
            if s != 0 and s != len(result) -1:
                front_overlap_ratio = (result[s-1]["end"] - result[s]["start"]) / (result[s]["end"] - result[s]["start"])
                rear_overlap_ratio = (result[s]["end"] - result[s+1]["start"]) / (result[s]["end"] - result[s]["start"])
                front_cluster = result[s-1]["cluster"]
                rear_cluster = result[s+1]["cluster"]

                if result[s]["cluster"].isdigit() or result[s]["cluster"] == "-1":
                    if front_cluster == rear_cluster and \
                        (front_overlap_ratio + rear_overlap_ratio) > 0.3:
                        result[s].update({"cluster": front_cluster})
        return result
    
def call(path: str, speaker_embedding: dict = None, *args):
    """
    @param path:                 str, Path to the input audio file.
    @param speaker_embedding:    dict, the filtered elements from speaker_anchor.json
    """

    # initializing instance and loading configuration file
    x = diarization(path)
    config = read_config(x.CONFIG_PATH)

    # output directory check
    dir_exist_check(x.OUTPUT_DIR)
    dir_exist_check(x.TEMP_FILE_DIR)

    # file type transforming 
    file_type_tansforming(x.INPUT_PATH, x.TEMP_FILE_PATH, x.audio_to_wav)

    # preprocessing 
    time_mapping_lst = x.preprocessing(x.TEMP_FILE_PATH, x.TEMP_FILE_PATH, if_vad = str2bool(config["preprocessing"]["vad"]),
                                    if_vad_keep_length=str2bool(config["preprocessing"]["vad_keep_length"]),
                                    if_preemphasis=str2bool(config["preprocessing"]["pre_emphasis"]))

    # speaker embedding
    if config["init"]["embed"].lower() == "ecapa":
        seg_arr, timestamp_lst = x.wav_segmentation(wav_path=x.TEMP_FILE_PATH, embedding_model=x.ECAPA_embedding,
                                                    window=config["segmentation"]["window"],
                                                    stride=config["segmentation"]["stride"],
                                                    batch_size=config["segmentation"]["batch"],
                                                    time_mapping_lst=time_mapping_lst)
        
        print("embedding array shape: {}".format(seg_arr.shape))
    else:
        raise ValueError("The embedding method does not exist.")
    
    # normalization
    if str2bool(config["init"]["normalizartion"]):
        seg_arr = l2_normalize(seg_arr)
        print("Normalization: True")
    else:
        print("Normalization: False")

    # speaker_similarity_compute
    if speaker_embedding:
        embedding_arr = np.empty((len(speaker_embedding["speaker"]), seg_arr.shape[-1]))
        for i in range(len(speaker_embedding["speaker"])):
            # print(speaker_embedding["speaker"][i]["output"]["embedding"])
            embedding_arr[i] =  np.load(speaker_embedding["speaker"][i]["output"]["embedding"])

        similarity_ratio = [np.max(cosine_similarity(s.reshape(1, -1), embedding_arr)) for s in seg_arr]
        most_similar_speaker = [speaker_embedding["speaker"][np.argmax(cosine_similarity(s.reshape(1, -1), embedding_arr))]["name"] for s in seg_arr]

    # dimension reduction
    kpca_params = config["dimension_reduction"].pop("kpca")
    print("dimension reduction method: {}".format(config["init"]["reduction"]))
    print("general_params: {}".format(config['dimension_reduction']))

    if config["init"]["reduction"].lower() == "pca":
        seg_arr = dimension_reduction(seg_arr, config["init"]["reduction"], **config['dimension_reduction'])

    elif config["init"]["reduction"].lower() == "kpca":
        print(f"kpca_params: {kpca_params}")
        seg_arr = dimension_reduction(seg_arr, config["init"]["reduction"], **{**config['dimension_reduction'], **kpca_params})

    else:
        raise ValueError("The dimension reduction method does not exist.")

    # cluster
    print("clustering method: {}".format(config["init"]["cluster"]))
    if config["init"]["cluster"].lower() == "hdbscan":
        cluster_result = clustering(seg_arr, config["init"]["cluster"], **config['clustering']['hdbscan'])
        
    elif config["init"]["cluster"].lower() == "ap":
        # `distance` is a parameter which isn’t from the original scikit-learn docs in customized_diarization_config.json.
        #  This parameter determines the kind of distance applied to calculating the similiarity of datapoints in affinitypropagation.
        distance = config["clustering"]["affinitypropagation"].pop("distance")

        if distance=="cosine":
            # if param `affinity` is set to `precomputed`, the input array should be similarity_matrix of the orginal array.
            # otherwise, if param `affinity` is set to `euclidean`, the input array should be the orginal array.
            similarity_matrix = cosine_similarity(seg_arr)
            similarity_median = 1 - np.median(similarity_matrix)
            print("similarity_median: {}".format(similarity_median))
            cluster_result = clustering(similarity_matrix, config["init"]["cluster"],
                                        **{**config['clustering']['affinitypropagation'],
                                        "affinity": "precomputed", "preference": similarity_median})
        else:
            cluster_result = clustering(seg_arr, config["init"]["cluster"],
                                        **{**config['clustering']['affinitypropagation'], 
                                       "affinity": "euclidean"})
            
    elif config["init"]["cluster"].lower() == "sc":
        cluster_result = clustering(seg_arr, config["init"]["cluster"], 
                                    **{**config['clustering']['spectralclustering'], 
                                    "n_clusters": config["init"]["speakers"]})
    else:
        raise ValueError("The clustering method does not exist.")
    
    # downstream processing
    clustered_seg_lst = x.assign_clusters(timestamp_lst, cluster_result)
    clustered_seg_lst = x.speaker_recognition(clustered_seg_lst, most_similar_speaker, similarity_ratio, config["init"]["threshold"])

    if not str2bool(config["preprocessing"]["vad_keep_length"]):
        clustered_seg_lst = timestamp_recover(time_mapping_lst, clustered_seg_lst)
        speaker_diarization_result = x.segmentation_annote_with_clusters(
            x.segmentation_annote_with_clusters(clustered_seg_lst, min_noise_ignore=200), 
            min_noise_ignore=0, report_seg_cnt = False
        )

    # speaker-cluster mapping
    if speaker_embedding:
        before_smoothing_len = 0
        after_smoothing_len = len(speaker_diarization_result)
        while before_smoothing_len != after_smoothing_len:
            print(f"before: {before_smoothing_len}, after: {after_smoothing_len}")
            before_smoothing_len = len(speaker_diarization_result)
            speaker_diarization_result = x.speaker_cluster_smoothing(speaker_diarization_result)
            speaker_diarization_result = x.segmentation_annote_with_clusters(speaker_diarization_result, min_noise_ignore=0, report_seg_cnt = False)
            after_smoothing_len = len(speaker_diarization_result)

    output_path = output(speaker_diarization_result, x.INPUT_PATH, x.OUTPUT_DIR, x.module)

    # clean temp folder
    for filename in os.listdir(x.TEMP_FILE_DIR):
        file_path = os.path.join(x.TEMP_FILE_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    return output_path

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", help="input audio path", dest="path", type=str, required=True)
    args = parser.parse_args()
    x = diarization(args.path)
    print("customized_diarization has initiailized.")


# Reference:
# on-premises ECAPA-TDNN 
# https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
# Step1: Download ECAPA-TDNN repo from huggingface via git and place it under your root directory.
# 1. make sure to pip install speechbrain version >= 1.0.0

# on-premises VAD
#Step 2 : Download seliro-vad repo from github via git and place it under your root directory.

# Note:
# 影響模型效果的重要參數
# 1. vad_keep_length in preprocessing()
# 2. window size, stride size in wav_segmentation
# 3. n_components in dimension reduction()
# 4. threshold in speaker_recognition()

# In traditional method, speaker diarization includes 4 part:
# 分割（segmentation，通常會去除沒有説話的片段）、
# 嵌入提取（embedding extraction）、
# 聚類（clustering）、
# 重分割（resegmentation）

# ECAPA-TDNN can not reach frame-level embedding, 
# resegmentation part can only achieve with MCFF and other ones can reach the frame-level embedding.
# https://www.sciencedirect.com/science/article/abs/pii/S088523082100111X