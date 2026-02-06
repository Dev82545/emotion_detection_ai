import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def evaluate_model(model, X_test, y_test, genders_test=None):
    model.eval()

    with torch.no_grad():
        outputs = model(X_test)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    y_true = y_test.cpu().numpy()

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro")

    print(f"Test Accuracy   : {acc:.4f}")
    print(f"Test Macro F1   : {f1:.4f}")

    # Pitch bias (optional but required in your spec)
    if genders_test is not None:
        male_idx = genders_test == "male"
        female_idx = genders_test == "female"

        male_f1 = f1_score(y_true[male_idx], preds[male_idx], average="macro")
        female_f1 = f1_score(y_true[female_idx], preds[female_idx], average="macro")

        print(f"Male Macro F1   : {male_f1:.4f}")
        print(f"Female Macro F1 : {female_f1:.4f}")

    return confusion_matrix(y_true, preds)
