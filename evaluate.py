import os
import sys
import numpy as np

# Allow imports from src/
sys.path.append(".")

from dataset_builder.data_features import load_and_trim, encode_labels
from mel_spectrogram import plot_mel
from dataset_builder.actor_splitting import actor_wise_split, build_dataset_from_actors
from config import DATA_DIR, OUTPUT_DIR



os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Step 1: Visual Analysis
# =========================
def visual_analysis():
    print("Performing visual analysis (Angry vs Sad)...")

    angry_file = None
    sad_file = None

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if not file.endswith(".wav"):
                continue

            emotion_id = file.split("-")[2]

            if emotion_id == "05" and angry_file is None:
                angry_file = os.path.join(root, file)
            elif emotion_id == "04" and sad_file is None:
                sad_file = os.path.join(root, file)

            if angry_file and sad_file:
                break

    y_angry, sr = load_and_trim(angry_file)
    y_sad, sr = load_and_trim(sad_file)

    plot_mel(y_angry, sr, "Angry Speech – Log-Mel Spectrogram")
    plot_mel(y_sad, sr, "Sad Speech – Log-Mel Spectrogram")

    print("Visual analysis complete.\n")


# =========================
# Step 2: Actor-wise Split
# =========================
def prepare_datasets():
    print("Performing actor-wise split...")

    train_actors, val_actors = actor_wise_split(DATA_DIR)

    print("Train actors:", train_actors)
    print("Validation actors:", val_actors, "\n")

    print("Building training dataset...")
    X_train, y_train = build_dataset_from_actors(DATA_DIR, train_actors)

    print("Building validation dataset...")
    X_val, y_val = build_dataset_from_actors(DATA_DIR, val_actors)

    print("\nShapes:")
    print("X_train:", X_train.shape)
    print("X_val  :", X_val.shape)

    return X_train, y_train, X_val, y_val


# =========================
# Step 3: Encode Labels
# =========================
def encode_and_save(X_train, y_train, X_val, y_val):
    print("\nEncoding labels...")

    y_train_enc, label_to_idx, idx_to_label = encode_labels(y_train)
    y_val_enc = [label_to_idx[l] for l in y_val]

    print("Label mapping:", label_to_idx)

    # CNN expects (N, C, H, W)
    X_train = X_train[:, None, :, :]
    X_val = X_val[:, None, :, :]

    print("\nFinal shapes:")
    print("X_train:", X_train.shape)
    print("X_val  :", X_val.shape)

    print("\nSaving processed data...")

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train_enc)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val_enc)

    print("Data saved to:", OUTPUT_DIR)


# =========================
# MAIN ENTRY POINT
# =========================
if __name__ == "__main__":
    print("\nStarting Phase-1 Processing\n")

    visual_analysis()

    X_train, y_train, X_val, y_val = prepare_datasets()

    encode_and_save(X_train, y_train, X_val, y_val)

    print("\nPhase-1 COMPLETE")
