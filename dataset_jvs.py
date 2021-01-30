import os

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig
import torch
from joblib import Parallel, delayed


phoneme2index = {
    "a"  :0,

    "I"  :1,
    "i"  :1,

    "U"  :2,
    "u"  :2,

    "e"  :3,

    "o"  :4,

    "N"  :5,
    "b"  :5,
    "by" :5,
    "m"  :5,
    "my" :5,
    "p"  :5,
    "py" :5,

    "f"  :6,
    "h"  :6,
    "hy" :6,
    "v"  :6,

    "cl" :7,
    "d"  :7,
    "g"  :7,
    "gy" :7,
    "k"  :7,
    "ky" :7,
    "n"  :7,
    "ny" :7,
    "s"  :7,
    "t"  :7,
    "y"  :7,
    "z"  :7,

    "ch" :8,
    "j"  :8,
    "sh" :8,
    "ts" :8,

    "r"  :9,
    "ry" :9,

    "w"  :10,

    "pau":11,
    "sil":11,
}


def listup_pathes(dir_jvs):
    fmt_wav = os.path.join(dir_jvs, 'jvs{:03d}', 'parallel100',
                                 'wav24kHz16bit', 'VOICEACTRESS100_{:03d}.wav')
    fmt_lab = os.path.join(dir_jvs, 'jvs{:03d}', 'parallel100', 'lab', 'mon',
                                                  'VOICEACTRESS100_{:03d}.lab')

    train = []
    val   = []
    for i_person in range(1, 101):
        for i_file in range(1, 101):
            path_wav = fmt_wav.format(i_person, i_file)
            path_lab = fmt_lab.format(i_person, i_file)
            if not os.path.exists(path_wav) or not os.path.exists(path_lab):
                continue
            if i_person < 80:
                train.append((path_wav, path_lab))
            else:
                val.append((path_wav, path_lab))
    
    return train, val


def load_lab(path):
    with open(path, 'r') as f:
        text = f.read()
    lines = text.strip().splitlines()
    phenomes = []
    for l in lines:
        start, end, phenome = l.strip().split(' ')
        start = int(float(start) * 16000)
        end   = int(float(end)   * 16000)
        index = phoneme2index[phenome]
        arr   = np.zeros(end - start, dtype=np.int64) + index
        phenomes.append(arr)
    phenomes = np.concatenate(phenomes)
    return phenomes


def load_wav(path):
    bps, data = wav.read(path)
    assert bps == 24000

    data = data.astype(np.float)
    data = sig.resample_poly(data, 2, 3)
    data /= np.max(np.abs(data))

    return data


def preprocess(path_wav, path_lab, n_win, n_hop, eps, n_frames=None):
    data_wav = load_wav(path_wav)

    frame_pos = np.arange(0, len(data_wav)-n_win, n_hop)
    center    = frame_pos + n_win//2
    window    = frame_pos.repeat(n_win).reshape(-1, n_win) \
              + np.arange(n_win).reshape(1, n_win)
    frame_wav = data_wav[window] * np.hanning(n_win).reshape(1, n_win)

    stft = np.fft.fft(frame_wav, axis=1)
    stft = stft.real**2 + stft.imag**2
    stft = np.log10(stft+eps)
    stft = stft[:,:n_win//2]

    if n_frames is None:
        return stft

    data_pho = load_lab(path_lab)
    if len(data_pho) < len(data_wav):
        dummy    = np.zeros(len(data_wav)-len(data_pho), dtype=np.int64) \
                 + data_pho[-1]
        data_pho = np.concatenate([data_pho, dummy])
    frame_pho = data_pho[center]
    
    frame_pos = np.arange(0, stft.shape[0]-n_frames, n_frames)
    window    = frame_pos.repeat(n_frames).reshape(-1, n_frames) \
              + np.arange(n_frames).reshape(1, n_frames)

    xs = np.transpose(stft[window], axes=(0, 2, 1))
    ts = frame_pho[window]

    return list(xs.astype(np.float32)), list(ts.astype(np.int64))


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, pathes, n_win, n_hop, eps, n_frames, n_skip):
        dsts = Parallel(n_jobs=-1, verbose=5)(
          [delayed(preprocess)(path_wav, path_lab, n_win, n_hop, eps, n_frames)
                                   for path_wav, path_lab in pathes[::n_skip]])
        self.xt = [xt for d in dsts for xt in zip(*d)]

    def __getitem__(self, index):
        x, t = self.xt[index]
        return x, t

    def __len__(self):
        return len(self.xt)
