import argparse
import torch
import numpy as np
import librosa

from cnn_model.model import EmotionCNN
from dataset_builder.data_features import load_and_trim
from config import (
    SAMPLE_RATE,
    N_MELS,
    FIXED_DURATION,
    EMOTION_LABELS
)

# --------------------------------------------------
# Feature extraction (MATCHES TRAINING PIPELINE)
# --------------------------------------------------
def extract_features(wav_path):
    # Load and trim audio (same as training)
    y, sr = load_and_trim(wav_path, SAMPLE_RATE)

    # Ensure fixed duration
    max_len = SAMPLE_RATE * FIXED_DURATION
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )

    # Log scale (used during training)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db


# --------------------------------------------------
# Emotion prediction
# --------------------------------------------------
def predict_emotion(wav_path, model_path, device):
    features = extract_features(wav_path)

    # Shape: (1, 1, n_mels, time)
    features = torch.tensor(features, dtype=torch.float32)
    features = features.unsqueeze(0).unsqueeze(0).to(device)

    # Load model
    model = EmotionCNN(num_classes=len(EMOTION_LABELS))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Inference
    with torch.no_grad():
        logits = model(features)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_idx = torch.argmax(probs).item()

    return EMOTION_LABELS[pred_idx], probs.cpu().numpy()


# --------------------------------------------------
# Command-line interface
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict emotion from a WAV file"
    )
    parser.add_argument(
        "--wav",
        type=str,
        required=True,
        help="Path to input .wav file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best_model.pth",
        help="Path to trained model weights (.pth)"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    emotion, probs = predict_emotion(
        args.wav,
        args.model,
        device
    )

    print(f"\nAudio File        : {args.wav}")
    print(f"Predicted Emotion : {emotion}\n")

    print("Class Probabilities:")
    for label, p in zip(EMOTION_LABELS, probs):
        print(f"  {label:10s}: {p:.4f}")
