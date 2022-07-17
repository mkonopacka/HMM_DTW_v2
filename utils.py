from xml.dom import ValidationErr
import numpy as np 

def accuracy(true: np.ndarray, pred: np.ndarray) -> float:
    if true.shape != pred.shape:
        raise Exception("Arrays must have equal shapes.")
    return np.mean(true == pred)
