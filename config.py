# This file contains all global parameters and constants used across the project.

# Audio parameters
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
MAX_LEN = 300
TOP_DB = 20

# Emotion mapping in RAVDESS Dataset
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

emotion_labels = [
        "neutral", "calm", "happy", "sad",
        "angry", "fearful", "disgust", "surprised"
    ]

# =========================
# Configuration
# =========================
DATA_DIR = "data/ravdess"
OUTPUT_DIR = "data/processed"