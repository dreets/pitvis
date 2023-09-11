# global imports
import pandas as pd
from sklearn import metrics

# strongly typed
from typing import List
from typing import Tuple
from pandas import DataFrame
from sklearn.preprocessing import MultiLabelBinarizer


def main():
    # ls_trues: List[List] = []  # ground-truth instruments per frame (2 integer values)
    # ls_preds: List[List] = []  # prediction instruments per frame (1 or 2 integer values)
    # flt_evaluation_metric: float = calculate_insts_evaluation_metric(ls_trues=ls_trues, ls_preds=ls_preds)
    pass


def calculate_insts_evaluation_metric(ls_trues: List[List], ls_preds: List[List]) -> float:
    """Calculate the instrument evaluation metric from a ground-truth list and prediction list."""
    df_trues_encoded, df_preds_encoded = clean_insts(ls_trues=ls_trues, ls_preds=ls_preds)
    flt_f1_score: float = metrics.f1_score(
        y_true=df_trues_encoded,
        y_pred=df_preds_encoded,
        average="macro",
        zero_division=1
    )
    return flt_f1_score


def clean_insts(ls_trues: List[List], ls_preds: List[List]) -> Tuple[DataFrame, DataFrame]:
    """Ensure input data is compatible with evaluation metric calculation."""
    ls_trues_pad, ls_preds_pad = check_insts_lists_are_compatible(ls_trues=ls_trues, ls_preds=ls_preds)
    ls_trues_no_bkg, ls_preds_no_bkg = remove_background_insts(ls_trues=ls_trues_pad, ls_preds=ls_preds_pad)
    df_trues_enc, df_preds_enc = hot_encode_insts(ls_trues=ls_trues_no_bkg, ls_preds=ls_preds_no_bkg)
    return df_trues_enc, df_preds_enc


def check_insts_lists_are_compatible(ls_trues: List[List], ls_preds: List[List]) -> Tuple[List[List], List[List]]:
    """Ensure truths and predictions are compatible, pad with (-1, -2) when necessary."""
    if len(ls_trues) != len(ls_preds):
        print(f"Lengths of truths ({len(ls_trues)}) and preds ({len(ls_preds)}) are not equal.")
        raise SystemExit
    for int_index, ls_true in enumerate(ls_trues):
        if len(ls_true) != 2:
            print(f"Lengths of truths at index={int_index} is {len(ls_true)}!=2.")
            raise SystemExit

    ls_preds_padded: List[List] = []
    for int_index, ls_pred in enumerate(ls_preds):
        if len(ls_pred) == 0:
            ls_pred_padded: List = [-1, -2]
        elif len(ls_pred) == 1:
            ls_pred_padded: List = [ls_pred[0], -2]
        elif len(ls_pred) == 2:
            ls_pred_padded: List = ls_pred
        else:
            print(f"Lengths of truths at index={int_index} is {len(ls_pred)}>2.")
            raise SystemExit
        ls_preds_padded.append(ls_pred_padded)
    return ls_trues, ls_preds_padded


def remove_background_insts(ls_trues: List[List], ls_preds: List[List]) -> Tuple[List[List], List[List]]:
    """Remove background class (-1), as defined by the the ground-truth."""
    df_trues: DataFrame = pd.DataFrame(ls_trues, columns=["inst1", "inst2"])
    df_preds: DataFrame = pd.DataFrame(ls_preds, columns=["inst3", "inst4"])
    df_trues_preds: DataFrame = pd.concat([df_trues, df_preds], axis=1)

    # check which column contains the "out_of_patient" class -1 (usually 'inst1') and remove those frames
    int_background_inst1: int = df_trues_preds[df_trues_preds["inst1"] == -1]["inst1"].count()
    int_background_inst2: int = df_trues_preds[df_trues_preds["inst2"] == -1]["inst2"].count()
    if int_background_inst1 > int_background_inst2:
        df_trues_preds_no_background: DataFrame = df_trues_preds[df_trues_preds["inst1"] != -1]
    else:
        df_trues_preds_no_background: DataFrame = df_trues_preds[df_trues_preds["inst2"] != -1]

    ls_trues_no_background: List[List] = df_trues_preds_no_background[["inst1", "inst2"]].to_numpy().tolist()
    ls_preds_no_background: List[List] = df_trues_preds_no_background[["inst3", "inst4"]].to_numpy().tolist()
    return ls_trues_no_background, ls_preds_no_background


def hot_encode_insts(ls_trues: List[List], ls_preds: List[List]) -> Tuple[DataFrame, DataFrame]:
    """Hot encode the both ground-truth and predictions."""
    mlb = MultiLabelBinarizer()
    df_trues = pd.Series(ls_trues)
    df_preds = pd.Series(ls_preds)
    df_trues_encoded: DataFrame = pd.DataFrame(mlb.fit_transform(df_trues), columns=mlb.classes_, index=df_trues.index)
    df_preds_encoded: DataFrame = pd.DataFrame(mlb.fit_transform(df_preds), columns=mlb.classes_, index=df_preds.index)
    df_trues_encoded.pop(-2)
    df_preds_encoded.pop(-2)
    ls_range: List[int] = [int_x for int_x in range(19)]
    for int_inst in ls_range:
        if int_inst not in df_trues_encoded.columns.to_list():
            df_trues_encoded[int_inst] = [0] * len(df_trues_encoded)
    for int_inst in ls_range:
        if int_inst not in df_preds_encoded.columns.to_list():
            df_preds_encoded[int_inst] = [0] * len(df_trues_encoded)
    return df_trues_encoded, df_preds_encoded


if __name__ == "__main__":
    main()
