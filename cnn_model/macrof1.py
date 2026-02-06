import numpy as np
from sklearn.metrics import f1_score, classification_report


def compute_macro_f1(y_true, y_pred, idx_to_label=None, verbose=True):
    """
    Compute and display Macro F1 score.

    Args:
        y_true (array-like): Ground truth labels (encoded ints)
        y_pred (array-like): Predicted labels (encoded ints)
        idx_to_label (dict, optional): index -> emotion label mapping
        verbose (bool): Whether to print detailed report

    Returns:
        float: Macro F1 score
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    if verbose:
        print(f"\nMacro F1 Score: {macro_f1:.4f}")

        if idx_to_label is not None:
            target_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
            print("\nPer-class performance:")
            print(classification_report(
                y_true,
                y_pred,
                target_names=target_names,
                digits=4
            ))

    return macro_f1
