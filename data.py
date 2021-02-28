from pathlib import Path
import pandas as pd
import numpy as np


def train_val_split(df: pd.DataFrame, frac: float):
    train_samples = np.random.rand(df.shape[0]) < 0.8
    val_samples = ~train_samples
    
    df_train = df.iloc[train_samples]
    df_val = df.iloc[val_samples]
    return df_train, df_val


def reformat_data_to_feather(read_folder: Path, write_folder: Path):
    df_train = pd.read_csv(read_folder / "train.csv")
    df_test = pd.read_csv(read_folder / "test.csv")
    
    df_train.to_feather(write_folder / "train.feather")
    df_test.to_feather(write_folder / "test.feather")


def read_data(folder: Path):
    df_train = pd.read_feather(folder / "train.feather").set_index("id")
    df_test = pd.read_feather(folder / "test.feather").set_index("id")
    
    return df_train, df_test
