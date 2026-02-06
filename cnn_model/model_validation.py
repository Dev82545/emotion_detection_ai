from sklearn.model_selection import train_test_split
from collections import Counter

def stratified_80_10_10_split(X, y, random_state=42):
    """
    Returns stratified Train / Val / Test split.
    """
    # 80% train, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )

    # Split temp into 10% val, 10% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=random_state
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def print_split_stats(y_train, y_val, y_test):
    print("Train distribution:", Counter(y_train))
    print("Val distribution:", Counter(y_val))
    print("Test distribution:", Counter(y_test))
