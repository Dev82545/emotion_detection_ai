'''
This is the code which builds the dataset by loading the audio files, extracting features and encoding the labels.
'''
import os
import numpy as np
from tqdm import tqdm

from .data_features import load_and_trim, parse_emotion
from mel_spectrogram.mel_generator import generate_logmel
from .data_augmentation import augment_audio


def build_dataset(file_paths, config, augment=False):
    """
    file_paths : list of .wav file paths (already split)
    augment    : True ONLY for training data
    """

    X = []
    y = []

    for path in tqdm(file_paths):
        if not path.endswith(".wav"):
            continue

        # 1️⃣ label parsing (independent of augmentation)
        emotion = parse_emotion(os.path.basename(path))

        # 2️⃣ load audio ONCE
        audio, sr = load_and_trim(path)

        # 3️⃣ original mel
        mel = generate_logmel(audio, sr)
        X.append(mel)
        y.append(emotion)

        # 4️⃣ augmented ONCE (training only)
        if augment:
            audio_aug = augment_audio(audio, sr)
            mel_aug = generate_logmel(audio_aug, sr)
            X.append(mel_aug)
            y.append(emotion)

    return np.array(X), np.array(y)