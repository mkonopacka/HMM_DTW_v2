import numpy as np 
import pandas as pd

def accuracy(true: np.ndarray, pred: np.ndarray) -> float:
    if true.shape != pred.shape:
        raise Exception("Arrays must have equal shapes.")
    return np.mean(true == pred)

def find_colnames_with(df: pd.DataFrame, what: str = "pred") -> list[str]:
    return [col for col in df if what in col]