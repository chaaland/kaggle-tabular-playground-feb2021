import os
import pprint

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import Ridge

from data import read_data, train_val_split
from feature_eng import cross_features
from config import ExperimentConfig

pjoin = os.path.join

DF_TRAIN, DF_TEST = None, None
CONFIG = None


def ridge_objective(trial):
    global DF_TRAIN
    global CONFIG
    df_fit, df_val = train_val_split(DF_TRAIN, 0.8)
    encoders = fit_target_encoders(df_fit.drop("target"), df_fit["target"])
    
    feature_cols = [c for c in df_fit if c.startswith("cont")]
    
    x_train = df_fit[feature_cols].to_numpy()
    y_train = df_fit["target"].to_numpy()
    
    x_val = df_val[feature_cols].to_numpy()
    y_val = df_val["target"].to_numpy()
 
    alpha_start, alpha_end = CONFIG.get("experiment/training/alpha_range")
    model = Ridge(alpha=trial.suggest_loguniform('alpha', alpha_start, alpha_end))
    model = model.fit(x_train, y_train)
    
    y_hat = model.predict(x_val)
    residual = y_val.ravel() - y_hat.ravel()
    rmse = np.sqrt(np.mean(np.square(residual)))
    
    return rmse


def train(config: ExperimentConfig):
    global CONFIG, DF_TRAIN, DF_TEST
    CONFIG = config
    df_train, df_test = read_data(config.get("experiment/data_path"))

    model_family = config.get("experiment/model/family")
    experiment_name = config.get("experiment/meta/name")
    study_name =  pjoin(model_family, experiment_name)  # Unique identifier of the study.
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)

    n_trials = config.get("experiment/training/n_optuna_trials")
    study.optimize(ridge_objective, n_trials=n_trials)

    print('Number of finished trials:', len(study.trials))
    print('Best trial: ') 
    pprint.pprint(study.best_trial.params)
    print(f'Best rmse: {study.best_trial.value:.6f}')
