'''
This is done so that the code doesn't start to cheat during the learning process
and identifies emotions and not actors.
'''

import os
import numpy as np
from tqdm import tqdm

from .data_features import load_and_trim, parse_emotion
from mel_spectrogram.mel_generator import extract_logmel
from .data_augmentation import augment_audio


def get_actor_dirs(data_dir: str):
    """
    Return sorted list of actor directories.
    """
    actors = []
    for d in os.listdir(data_dir):
        if d.startswith("Actor_"):
            actors.append(d)
    
    actors.sort()
    return actors


def actor_wise_split(data_dir: str, train_ratio: float = 0.8):
    """
    Split actors into train and validation sets.
    """
    actors = get_actor_dirs(data_dir)
    n_train = int(len(actors) * train_ratio)

    train_actors = actors[:n_train]
    val_actors = actors[n_train:]

    return train_actors, val_actors


def build_dataset_from_actors(data_dir: str, actor_list: list, augment: bool = False):
    X, y, genders = [], [], []

    for actor in tqdm(actor_list, desc="Building dataset"):
        actor_dir = f"Actor_{actor:02d}"
        actor_path = os.path.join(data_dir, actor_dir)

        gender = "male" if actor % 2 == 1 else "female"

        for file in os.listdir(actor_path):
            if not file.endswith(".wav"):
                continue

            path = os.path.join(actor_path, file)
            emotion = parse_emotion(file)

            audio, sr = load_and_trim(path)
            mel = extract_logmel(audio, sr)
            X.append(mel)
            y.append(emotion)

            if augment:
                audio_aug = augment_audio(audio, sr)
                features_aug = extract_logmel(audio_aug, sr)
                X.append(features_aug)
                y.append(emotion)
                genders.append(gender)

    return np.array(X), np.array(y), np.array(genders)

