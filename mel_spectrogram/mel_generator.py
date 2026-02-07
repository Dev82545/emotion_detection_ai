import numpy as np
import librosa
from config import SAMPLE_RATE, N_MELS, HOP_LENGTH, MAX_LEN

def extract_logmel(y, sr, max_len):
    """
    Convert waveform to fixed-size Log-Mel Spectrogram.
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Pad / trim time axis
    if log_mel.shape[1] < max_len:
        pad_width = max_len - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)))
    else:
        log_mel = log_mel[:, :max_len]

    return log_mel
