# local imports
from helper_scripts.evaluation_instruments import calculate_insts_evaluation_metric
from evaluation_steps import calculate_steps_evaluation_metric

# strongly typed
from typing import List


def main():
    # ls_trues_steps: List[int] = []  # ground-truth steps per frame (1 integer value)
    # ls_preds_steps: List[int] = []  # prediction steps per frame (1 integer value)
    # ls_trues_insts: List[List] = []  # ground-truth instruments per frame (2 integer values)
    # ls_preds_insts: List[List] = []  # prediction instruments per frame (1 or 2 integer values)
    # flt_evaluation_metric: float = calculate_multitask_evaluation_metric(
    #     ls_trues_steps=ls_trues_steps,
    #     ls_preds_steps=ls_preds_steps,
    #     ls_trues_insts=ls_trues_insts,
    #     ls_preds_insts=ls_preds_insts,
    # )
    pass


def calculate_multitask_evaluation_metric(
        ls_trues_steps: List[int],
        ls_preds_steps: List[int],
        ls_trues_insts: List[List],
        ls_preds_insts: List[List],
) -> float:
    """Calculate the multitask evaluation metric from a ground-truth list and prediction list."""
    flt_metric_steps: float = calculate_steps_evaluation_metric(ls_trues=ls_trues_steps, ls_preds=ls_preds_steps)
    flt_metric_insts: float = calculate_insts_evaluation_metric(ls_trues=ls_trues_insts, ls_preds=ls_preds_insts)
    flt_metric: float = (flt_metric_steps + flt_metric_insts) / 2
    return flt_metric


if __name__ == "__main__":
    main()
