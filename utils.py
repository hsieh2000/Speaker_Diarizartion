import torch
import onnxruntime
import librosa
import soundfile as sf
import noisereduce as nr
from sklearn.preprocessing import normalize
from sklearn.decomposition import KernelPCA, PCA
from sklearn.cluster import AffinityPropagation, HDBSCAN, SpectralClustering
import numpy as np
import json
import os
from pathlib import Path
import time

# you might need ffmpeg manually to execute moviepy.VideoFileClip. See the links below for installation instructions.
# https://www.geeksforgeeks.org/installation-guide/how-to-install-ffmpeg-on-windows/
# https://the-walking-fish.com/p/install-ffmpeg-on-windows/
from moviepy import VideoFileClip

# To import magic:
# For windows, install python-magic-bin 0.4.14(`libmagic` will be installed automatically). 
# For others, install python-magic and make sure `libmagic` has been installed in your system.
# If installing wrong package, it will possible cause Error: Segmentation fault (especially on windows). 
import magic 

def video_to_audio(input_path: str | Path, output_path: str | Path) -> str|Path:
    """
    This function convert a video file to a WAV format audio file.

    Inputs:
        @param input_path:  Path to the input video file.
        @param output_path: Path to the output audio file.

    Return:
        str|Path, the path to the output audio file.
    """
    video = VideoFileClip(input_path)
    file_exist_check(output_path, video.audio.write_audiofile)
    # video.audio.write_audiofile(output_path)
    return output_path

def valid_input_type_check(path: str | Path) -> str|None|bool:
    """
    This function check if the input file is valid (a video/audio file).

    Inputs:
        @param path: Path to the input file.
    
    Return:
        str|None|bool, If the input is a valid file, the function will return the file type. 
    """
    try:
    # -------------------------------------------
        # mime_type = magic.from_file(path, mime=True) # this script will fail when the path includes non-ASCII characters
    # -------------------------------------------
        with open(path, 'rb') as f:
            file_bytes = f.read()
            mime_type = magic.from_buffer(file_bytes, mime=True)
    # -------------------------------------------

        if mime_type.startswith('video/'):
            return "video"
        elif mime_type.startswith('audio/'):
            return "audio"
        else:
            print("unsupported input file type")
            return None
        
    except magic.MagicException as e:
        print(f"Error detecting MIME type: {e}")
        return False
    
def dir_exist_check(path: str|Path) -> bool:
    """
    This function check the existence of the path to a directory.
    If the path exists, return True, else return False.

    Inputs:
        @param path: Path to the directory.

    Return:
        bool
    """
    if os.path.exists(path):
        return True
    else:
        print(f"output directory \"{path}\" doesn't exist.")
        print(f"creating directory \"{path}\"...")
        os.mkdir(path)
        return False

def file_exist_check(path: str|Path, callback: object, *args) -> bool:
    """
    This function check the existence of the path to a file.
   
    Inputs:
        @param path:    Path to the file.
        @callback:      Customized callback fuction.

    Return:
        bool
    """
    if os.path.exists(path):
        return True
    else:
        callback(*args, path)
        return False

def read_config(path: str|Path) -> dict:
    """
    This function load the configuration file and convert it to dictionary format.
    Inputs:
        @param path: Path to the config file.
    Return:
        dict, the dictionary of the configuration file.
    """
    with open(path, "r") as f:
        return json.loads(f.read())

def str2bool(v: str) -> bool:
    """
    This function convert a boolean in string format to an actual boolean. 
    Inputs:
        @param path: string.
    Return:
        bool
    """
    if isinstance(v, bool):
        return v
    elif v.lower() == 'true':
        return True
    elif v.lower() ==  'false':
        return False

def vad(input_path: str | Path, output_path: str | Path, vad_module_path: str | Path, keep_length: bool = True) -> list[dict]|None:
    """
    This function is doing Voice Activity Detection.

    Inputs:
        @param input_path:        Path to the input audio file.
        @param vad_module_path:   Path to the local Silero-VAD repository., the repo's link is https://github.com/snakers4/silero-vad.
        @param output_path:  Path to the output audio file will be saved.
        @param keep_length:       Determines whether to keep the original length of time. If set to False, the output will be a concatenated audio file containing only the voice segments.
    Return:
        list[dict]|None, if keep_length is False, the function will return a list[dict] including the original - trimmed time mapping information. Otherwise, return None.
    """
    # 載入模型
    model, utils = torch.hub.load(
        repo_or_dir=vad_module_path,             # 包含 hubconf.py 的本地目錄
        model='silero_vad',
        source='local',                          # 這告訴 Hub 從 local 而非 github 拉 code
        trust_repo=True,                         # 若第一次使用，避免 prompt 阻塞
        force_reload=True,
        onnx=True
    )

    # 載入工具庫
    (get_speech_timestamps,
    save_audio,
    read_audio,
    VADIterator,
    collect_chunks) = utils

    # 讀取音檔
    wav = read_audio(input_path, sampling_rate=16000)

    # 取得音檔的時間區段
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
    # print(speech_timestamps)

    if keep_length:
        output = torch.zeros_like(wav)  # 原長度，全靜音
        for seg in speech_timestamps:
            output[seg['start']:seg['end']] = wav[seg['start']:seg['end']]
        save_audio(output_path, output, sampling_rate=16000)

    else:
        # 儲存僅有人聲的區段
        output = collect_chunks(speech_timestamps, wav)
        save_audio(output_path, output, sampling_rate=16000)
        new_start = 0.0
        time_mapping_lst = []
        for t in speech_timestamps:
            new_end = (new_start + (t["end"] - t["start"])/16000)
            time_mapping_lst.append(
                {
                    "origin start": t["start"]/16000,
                    "origin end": t["end"]/16000,
                    "new start": new_start,
                    "new end": new_end,
                } 
            )
            new_start = new_end

        return time_mapping_lst
    return None
            
def pre_emphasis(input_path: str | Path ="./temp/speech_only.wav" , output_path: str | Path = "./temp/speech_only.wav", sample_rate: int = 16000) -> np.ndarray:
    """
    This function is doing pre-emphasis.
    Inputs:
        @param input_path: Path to the input file.
        @param output_path:  Path to the output file.
    Return:
        np.ndarray, the wav in numpy array format.
    """
    y, sr = librosa.load(input_path, sr=sample_rate)
    # wav = librosa.effects.preemphasis(y)
    wav = nr.reduce_noise(y= y, sr=sr)
    wav = librosa.effects.preemphasis(wav)
    sf.write(output_path, wav, sr)
    
    return wav

def l2_normalize(arr: np.array) -> np.ndarray:
    """
    This function is doing the L2 normalization, which will constraint the length of a vector to 1.
    Inputs:
        @param arr: The speaker embedding of the audio file, the size of the embedding should be (num_of_segs, dim_of_embedding).
    Return:
        np.ndarray, the normalized vector.
    """
    return normalize(arr, norm='l2', axis=1)

def DR_param_selector(criteria, **kwargs) -> dict:
    """
    This function determines the criteria for dimension reduction process (ratio / n_components).

    Inputs:
        @param criteria: The criteria dimension reduction process, it can be either "ratio" or "n_components".
        @param **kwargs: The params in the dimension_reduction part of the configuration file.
    Reutrn:
        dict, The params in the dimension_reduction part of the configuration file in dictionary format.
    """
    n_components = kwargs.pop('n_components')
    if criteria == "n_components":
        return {**kwargs, "n_components": n_components}
    return kwargs

def dimension_reduction(embedding, method, **kwargs) -> np.ndarray:
    """
    This function is doing dimension reduction (PCA/KPCA).

    Inputs:
        @param embedding:     (required) The speaker embedding of the audio file, the size of the embedding should be (num_of_segs, dim_of_embedding).
        @param method:        (required) The decomposition method applied to the speaker embedding.
        @param n_components:  (optional) The number of compoemts that will yield after dimension reduction.
        @param ratio:         (optional) The expected explained variance ratio uses to determine how many principal components to retain.
        @param criteria:      (optional) Specifies whether to apply n_components or ratio during dimensionality reduction.
        @param kernel:        (optional) The kernel type for KPCA, required only if the method parameter is set to 'KPCA'.
        @param gamma:         (optional) The gamma for KPCA, required only if the method parameter is set to 'KPCA'.
    Return:
        np.ndarray, the numpy array processed by dimension reduction.
    """
    criteria = kwargs.pop('criteria')
    ratio = kwargs.pop('ratio')
    # n_components = kwargs.pop('n_components')

    if method.lower() == 'pca':
        param = DR_param_selector(criteria, **kwargs)
        pca = PCA(**param)
        pca_result = pca.fit_transform(embedding)
    elif method.lower() == 'kpca':
        param = DR_param_selector(criteria, **kwargs)
        kpca = KernelPCA(**param)
        pca_result = kpca.fit_transform(embedding)
    
    if criteria == "ratio":
        evr = get_explained_variance_ratio(pca_result)
        return np.array(pca_result[:, :int(np.where(evr > ratio)[0][0])])
    elif criteria == "n_components":
        return np.array(pca_result)

def get_explained_variance_ratio(pca_result) -> np.ndarray:
    """
    This function calculates the cumulative explained variance ratio for the given principal components.
    Inputs:
        @param pca_result:    The output of dimension_reduction().
    Return:
        np.ndarray, the array of the cumulative explained variance ratio.
    """
    explained_variance = np.var(pca_result, axis=0)  # 每個成分的變異量
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    cumulative_ratio = np.cumsum(explained_variance_ratio)
    return cumulative_ratio

def clustering(embedding, method:str, **kwargs) -> np.ndarray:
    """
    This function is doing clustering.
    Inputs:
        @param embedding:         (required) The speaker embedding of the audio, the size of the embedding should be (num_of_segs, dim_of_embedding).
        @param method:            (required) str,   The clustering method applied to the embedding.
        @param min_cluster_size:  (optional) int,   The minimum number of samples in a group for that group to be considered a cluster; groupings smaller than this size will be left as noise, required only if the method parameter is set to 'HDBSCAN'.
        @param min_samples:       (optional) int,   The parameter k used to calculate the distance between a point x_p and its k-th nearest neighbor. When None, defaults to min_cluster_size, required only if the method parameter is set to 'HDBSCAN'.
        @param metric:            (optional) str,   The metric to use when calculating distance between instances in a feature array, required only if the method parameter is set to 'HDBSCAN'.
        @param alpha:             (optional) float, A distance scaling parameter as used in robust single linkage, required only if the method parameter is set to 'HDBSCAN'.
        @param random_state:      (optional) int,   Pseudo-random number generator to control the starting state. Use an int for reproducible results across function calls, required only if the method parameter is set to 'AffinityPropagation'.
        @param convergence_iter:  (optional) int,   Number of iterations with no change in the number of estimated clusters that stops the convergence, required only if the method parameter is set to 'AffinityPropagation'.

    Return:
        np.ndarray, the array of the clustering result.
    """
    if method.lower() == 'hdbscan':
        # print('HDBSCAN: \n min_cluster_size: {} \n min_samples: {} \n metric: {} \n alpha: {}'.format(kwargs.get('min_cluster_size'), kwargs.get("min_samples"), kwargs.get('metric'), kwargs.get('alpha')))
        print('HDBSCAN:')
        print(kwargs)        
        clustering = HDBSCAN(**kwargs).fit_predict(embedding)
    elif method.lower()== 'ap':
        print(f'affinitypropagation:')
        print(kwargs)        
        clustering = AffinityPropagation(**kwargs).fit_predict(embedding)
    elif method.lower()=="sc":
        print(f'spectralclustering:')
        print(kwargs)        
        clustering = SpectralClustering(**kwargs).fit_predict(embedding)
    else:
        clustering = None

    return clustering

def gpu_available():
    # print(torch.cuda.is_available())
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    return device

def timestamp_recover(time_mapping_lst:list[dict], clustered_seg_lst:list[dict]) -> list[dict]:
    """
    This function recovers the trimmed timeline to the original timeline.
    Inputs:
        @param time_mapping_lst:    List[dict], each containing the original and trimmed start/end timestamps corresponding to segments within the audio file generated by vad().
        @param clustered_seg_lst:   List[dict], each containing the trimmed start/end timestamps and the clusters corresponding to segments within the audio file generated by assign_clusters() in customized_diarization.py.
    Return:
        list[dict], Keeping the same output format as the output of assign_clusters() in customized_diarization.py but the ["start"] and ["end"] are changed.
    """
    for seg in clustered_seg_lst:
        start_modified = False
        end_modified = False
        for i in range(len(time_mapping_lst)):
            current_time_seg = time_mapping_lst[i]
            prev_time_seg = time_mapping_lst[i-1] if i != 0 else {"new start": -1, "new end": -1}
            if (prev_time_seg["new start"] < seg["start"]) and (seg['start'] <= current_time_seg["new start"]) and not start_modified:
                seg.update({
                    "start": round((seg["start"]- current_time_seg["new start"])/
                                    ( current_time_seg["new end"]-current_time_seg["new start"])*
                                    ( current_time_seg["origin end"]- current_time_seg["origin start"])+
                                    current_time_seg["origin start"]
                                , 2)
                })
                start_modified = True

            if (prev_time_seg["new end"] < seg['end']) and (seg['end'] <= current_time_seg["new end"]) and not end_modified:
                seg.update({
                    "end": round((seg["end"]- current_time_seg["new start"])/
                                    ( current_time_seg["new end"]-current_time_seg["new start"])*
                                    ( current_time_seg["origin end"]-current_time_seg["origin start"])+
                                    current_time_seg["origin start"]
                                , 2)
                })
                end_modified = True   
    return clustered_seg_lst

def output(lst_of_dict: list[dict], input_file_path: str|Path, output_directory_path: str|Path, module: str) -> str|Path:
    """
    This function outputs the result to the assigned path.
    Inputs:
        @param lst_of_dict:             List[dict], the output of segmentation_annote_with_clusters() in customized_diarization.py.
        @param input_file_path:         str|Path,   Path to the input file.
        @param output_directory_path:   str|Path,   Path to the output directory.
    Return:
        str|Path, the output path.
    """
    filename, extension = os.path.basename(input_file_path).split('.')
    output_dict = {
        # "filename": os.path.basename(self.INPUT_PATH),
        "filename": f"{filename}.{extension}",
        "diarization": []
        }

    output_dict['diarization'] = lst_of_dict
    output_path = os.path.join(output_directory_path, f'{filename}_{module}_{time.time_ns()}.json')
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=4, ensure_ascii=False)
    return output_path

def file_type_tansforming(input_path: str|Path, output_path: str|Path, callback: object) -> bool:
    """
    This function performs different operations based on the file type.
    Inputs:
        @param input_path:  str|Path,   Path to the input file.
        @param output_path: str|Path,   Path to the output directory.
        @param callback:    str|Path,   customized callback function.
    Return:
        bool
    """
    file_type = valid_input_type_check(input_path)
    if file_type == "video":
        video_to_audio(input_path, output_path)
    elif file_type == "audio":
        fec = file_exist_check(output_path, callback, input_path)
        if not fec:
            print(f"copy {input_path} to {output_path}.")
    elif file_type == False:
        raise TypeError("Input must be the str of a file path.") 
    elif file_type is None:
        raise ValueError("Not a video or audio file.")
    return True