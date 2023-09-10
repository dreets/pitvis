# global imports
from itertools import groupby
from numpy import zeros
from sklearn import metrics

# strongly typed
from numpy import ndarray
from typing import List
from typing import Tuple


def main():
    # ls_trues: List[int] = []  # ground-truth steps per frame (1 integer value)
    # ls_preds: List[int] = []  # prediction steps per frame (1 integer value)
    # flt_evaluation_metric: float = calculate_steps_evaluation_metric(ls_trues=ls_trues, ls_preds=ls_preds)
    pass


def calculate_steps_evaluation_metric(ls_trues: List[int], ls_preds: List[int]) -> float:
    """Calculate the step evaluation metric from a ground-truth list and prediction list."""
    ls_trues_clean, ls_preds_clean = clean_steps(ls_trues=ls_trues, ls_preds=ls_preds)
    flt_f1_score: float = metrics.f1_score(y_true=ls_trues_clean, y_pred=ls_preds_clean, average="macro")
    flt_edit_score: float = calculate_edit_score(ls_trues=ls_trues_clean, ls_preds=ls_preds_clean, bl_norm=True)
    flt_metric: float = (flt_f1_score + flt_edit_score) / 2
    return flt_metric


def clean_steps(ls_trues: List[int], ls_preds: List[int]) -> Tuple[List[int], List[int]]:
    """Ensure input data is compatible with evaluation metric calculation."""
    ls_trues, ls_preds = check_steps_lists_are_compatible(ls_trues=ls_trues, ls_preds=ls_preds)
    ls_trues_cln, ls_preds_cln = remove_background_steps(ls_trues=ls_trues, ls_preds=ls_preds)
    return ls_trues_cln, ls_preds_cln


def check_steps_lists_are_compatible(ls_trues: List[int], ls_preds: List[int]) -> Tuple[List[int], List[int]]:
    """Ensure truths and predictions are compatible."""
    if len(ls_trues) != len(ls_preds):
        print(f"Lengths of truths ({len(ls_trues)}) and preds ({len(ls_preds)}) are not equal.")
        raise SystemExit
    return ls_trues, ls_preds


def remove_background_steps(ls_trues: List[int], ls_preds: List[int]) -> Tuple[List[int], List[int]]:
    """Remove classes that are not to be used in the the calculation of the evaluation metric (-1, 11, 13)."""
    ls_trues_clean: List[int] = []
    ls_preds_clean: List[int] = []
    for int_index, int_truth in enumerate(ls_trues):
        if int_truth not in [-1, 11, 13]:
            ls_trues_clean.append(int_truth)
            ls_preds_clean.append(ls_preds[int_index])
    return ls_trues_clean, ls_preds_clean


def calculate_edit_score(ls_trues: List[int], ls_preds: List[int], bl_norm: bool = True) -> float:
    """The distance between two sequences, as defined in https://en.wikipedia.org/wiki/Levenshtein_distance.
    bl_norm=True normalises the edit score for each sequence by the max number of segments in either the ground-truth or
    the prediction.
    This code is based on: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py.
    """
    ls_trues_sequence: List[int] = [key for key, _group in groupby(ls_trues)]
    ls_preds_sequence: List[int] = [key for key, _group in groupby(ls_preds)]

    int_row: int = len(ls_preds_sequence)
    int_col: int = len(ls_trues_sequence)
    d_matrix: ndarray = zeros([int_row + 1, int_col + 1], float)
    for int_i in range(int_row + 1):
        d_matrix[int_i, 0] = int_i
    for int_i in range(int_col + 1):
        d_matrix[0, int_i] = int_i
    for int_j in range(1, int_col + 1):
        for int_i in range(1, int_row + 1):
            if ls_trues_sequence[int_j - 1] == ls_preds_sequence[int_i - 1]:
                d_matrix[int_i, int_j] = d_matrix[int_i - 1, int_j - 1]
            else:
                d_matrix[int_i, int_j] = 1 + min(
                    d_matrix[int_i - 1, int_j],
                    d_matrix[int_i, int_j - 1],
                    d_matrix[int_i - 1, int_j - 1]
                )
    if bl_norm:
        flt_score: float = (1 - d_matrix[-1, -1] / max(int_row, int_col))
    else:
        flt_score: float = d_matrix[-1, -1]
    return flt_score


if __name__ == "__main__":
    main()
