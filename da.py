import os
import random
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import shift
import librosa
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
random.seed(cfg.seed)

### Data augmentation

# (Shift Audio)：
def shift_audio(audio):
    audio_duration = len(audio)
    max_shift_amount = audio_duration * cfg.shift_max
    min_shift_amount = audio_duration * cfg.shift_min
    shift_amount = random.uniform(min_shift_amount, max_shift_amount)
    shifted_audio = shift(audio, shift_amount)

    adict["shift_amount"].append("{:.2f}".format(shift_amount))
    return shifted_audio

# (Change Tempo)：
def change_tempo(audio):
    tempo_factor = random.choice(np.arange(cfg.tempo_min, cfg.tempo_max, cfg.tempo_stride))
    audio_with_changed_tempo = librosa.effects.time_stretch(y=audio, rate=tempo_factor)

    adict["tempo_factor"].append("{:.2f}".format(tempo_factor))
    return audio_with_changed_tempo

# （Add noise)：
def add_noise(audio):
    noise_type = random.choice(['gaussian', 'uniform', 'white'])
    noise_factor = random.choice(np.arange(cfg.noise_min_factor, cfg.noise_max_factor, cfg.noise_stride))

    min_time = int(len(audio) * cfg.noise_time_min)
    max_time = int(len(audio) * cfg.noise_time_max)
    noise_length = random.randint(min_time, max_time)
    current_noise_length = noise_length

    if noise_type == 'gaussian':
        noise = np.random.randn(current_noise_length)
        noise = noise_factor * np.max(audio) * noise
    elif noise_type == 'uniform':
        noise = np.random.uniform(-1, 1, current_noise_length)
        noise = noise_factor * np.max(audio) * noise

    elif noise_type == 'white':
        noise = np.random.normal(0, 1, current_noise_length)
        noise = noise_factor * np.max(audio) * noise

    start_time = random.randint(0, len(audio) - current_noise_length)
    end_time = start_time + current_noise_length
    num_samples = end_time - start_time
    sample_points = np.linspace(start_time, end_time, num_samples, endpoint=False, dtype=int)
    audio_with_noise = audio.copy()
    audio_with_noise[sample_points] += noise

    adict["noise_type"].append(format(noise_type))
    adict["noise_factor"].append("{:.2f}".format(noise_factor))
    return audio_with_noise

# Pitch Shifting
def pitch_shift(sig):
    n_steps = random.choice(np.arange(cfg.pitch_min, cfg.pitch_max, cfg.pitch_stride))
    trans_sig = librosa.effects.pitch_shift(y=sig, sr=cfg.fs, n_steps=n_steps)
    adict["pitch_steps"].append("{:.2f}".format(n_steps))
    return trans_sig

def sig_index_(cur_samples, tar_samples, sig):
    x_original = np.linspace(0, 1, cur_samples)
    x_target = np.linspace(0, 1, tar_samples)
    sig = np.interp(x_target, x_original, sig)
    return sig

def dur_process(sig):
    cur_samples = int(len(sig))
    tar_samples = cfg.wav_time * cfg.fs
    if cur_samples < tar_samples:
        sig = sig_index_(cur_samples, tar_samples, sig)
    elif cur_samples == tar_samples:
        pass
    else:
        sig = sig[:cfg.wav_time * cfg.fs]
    return sig.astype(np.float32)



file_dir = f"{cfg.wav_dir}/train_data"
output_dir = f"{cfg.wav_dir}/augmix{cfg.da_multi}_data"  # augmix{cfg.da_multi}  aug1_data
adict = {"shift_amount":[],"tempo_factor":[],"noise_factor":[],"noise_type":[], "pitch_steps":[]}

for class_folder in os.listdir(file_dir):
    #if class_folder == 'hungry':
        #da_multi = 3
    #else:
        #da_multi = 10
    class_folder_path = os.path.join(file_dir, class_folder)

    daclass_folder_path = os.path.join(output_dir, class_folder)
    if not os.path.exists(daclass_folder_path): os.makedirs(daclass_folder_path)

    file_list = os.listdir(class_folder_path)
    tar_filenumbs = len(file_list)
    random_files = random.sample(file_list, tar_filenumbs)

    counts = 0
    for i in range(tar_filenumbs*cfg.da_multi):
        selected_file = random_files[i % tar_filenumbs]
        tar_name = selected_file.replace('.wav',f'_da{counts}.wav')
        tar_path = os.path.join(daclass_folder_path, tar_name)
        file_path = os.path.join(class_folder_path, selected_file)
        sig, _ = librosa.load(file_path, sr=None)
        
        methods_weights = [1.5, 1.5, 1.5,
                           2, 1, 2, 1]
        selected_method = random.choices(["shift_noise", "tempo_noise", "pitch_noise",
                                          "shift_tempo", 'shift_tempo_noise', "pitch_tempo",'pitch_tempo_noise'],
                                         methods_weights, k=1)[0]
        # shift_audio change_tempo add_noise pitch_shift
        if selected_method == "shift_noise":
            da_audio = add_noise(shift_audio(sig)).astype(np.float32)
        elif selected_method == "tempo_noise":
            da_audio = add_noise(change_tempo(sig)).astype(np.float32)
        elif selected_method == "pitch_noise":
            da_audio = add_noise(pitch_shift(sig)).astype(np.float32)
        elif selected_method == "shift_tempo":
            da_audio = change_tempo(shift_audio(sig)).astype(np.float32)
        elif selected_method == "shift_tempo_noise":
            da_audio = add_noise(change_tempo(shift_audio(sig))).astype(np.float32)
        elif selected_method == "pitch_tempo":
            da_audio = change_tempo(pitch_shift(sig)).astype(np.float32)
        elif selected_method == "pitch_tempo_noise":
            da_audio = add_noise(change_tempo(pitch_shift(sig))).astype(np.float32)

        da_audio = dur_process(da_audio)
        wavfile.write(tar_path, cfg.fs, da_audio)
        counts += 1

# adict = {"shift_amount":[],"tempo_factor":[],"noise_factor":[],"noise_type":[], "pitch_steps":[]}
msf = "shift: amount {}, value {} \n tempo: amount {}, value {} \n noise: amount {}, value {} \n " \
      "pitch: amount {}, value {} \n sr value: {} \n Total:{} "
'''
print(msf.format(
      len(adict["shift_amount"]),
      len(adict["tempo_factor"]),
      len(adict["noise_factor"]),
      len(adict["noise_type"]),
      len(adict["pitch_steps"]),))'''
