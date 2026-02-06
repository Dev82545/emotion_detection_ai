from src.split import build_dataset_from_actors
from src.validation import stratified_80_10_10_split, print_split_stats

# Path to your RAVDESS dataset root
DATASET_DIR = r"C:\Emotion_Detection_AI\data\ravdess-"   # adjust if needed

# RAVDESS actors: 1 to 24
ALL_ACTORS = list(range(1, 25))

# Build full dataset
X_all, y_all = build_dataset_from_actors(DATASET_DIR, ALL_ACTORS)

print(f"Total samples: {len(X_all)}")

# Stratified 80/10/10 split
X_train, y_train, X_val, y_val, X_test, y_test = stratified_80_10_10_split(
    X_all, y_all
)

# Print emotion distributions
print_split_stats(y_train, y_val, y_test)


