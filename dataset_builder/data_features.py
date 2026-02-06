'''
This file contains functions to extract features from the audio files and encode the emotion labels.
'''

import librosa
from config import SAMPLE_RATE, TOP_DB, EMOTION_MAP


def parse_emotion(filename: str) -> str:
    """
    Extract emotion label from RAVDESS filename.
    """
    emotion_id = filename.split("-")[2]
    return EMOTION_MAP[emotion_id]


def load_and_trim(path: str, sr: int = SAMPLE_RATE):
    """
    Load audio and remove leading/trailing silence.
    """
    y, sr = librosa.load(path, sr=sr)
    y_trimmed, _ = librosa.effects.trim(y, top_db=TOP_DB)
    return y_trimmed, sr

def encode_labels(labels):
    """
    Convert emotion string labels to integer labels.
    """
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    encoded_labels = [label_to_idx[label] for label in labels]

    return encoded_labels, label_to_idx, idx_to_label
