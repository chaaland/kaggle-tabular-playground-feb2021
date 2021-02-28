from pathlib import Path

import pandas as pd
import numpy as np


def save_preds(y_pred: np.ndarray, ids: np.ndarray, folder: Path, submission_name):
    submission_df = pd.DataFrame(data=y_pred, columns=["target"], index=ids).reset_index()
    submit_file = folder / f"{submission_name}.csv"
    submission_df.to_csv(submit_file, index=False)
    
    
def ensemble_submissions(submission_files: list):
    prediction_frames = []
    for submission_file in submission_files:
        pred_df = pd.read_csv(submission_file).set_index("id")
        prediction_frames.append(pred_df)
   
    prediction_df = pd.concat(prediction_frames, axis=0).mean(axis=0)
    return prediction_df
