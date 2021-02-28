import itertools as it
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def get_numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include=np.number).columns.tolist()


def get_categorical_cols(df: pd.DataFrame):
    return df.select_dtypes(include="object").columns.tolist()


def cross_string_features(string_df: pd.DataFrame, n_crossings: int=2):
    string_df = string_df.copy()
    cross_features = []

    for cols in it.combinations(string_df, r=n_crossings):
        cross_feature = ",".join(cols)
        string_df[cross_feature] = ""
        for i, c in enumerate(cols):
            sep = "" if i == 0 else ","
            string_df[cross_feature] += sep + string_df[c]
        cross_features.append(cross_feature)
    
    string_cross_df = string_df[cross_features]
    return string_cross_df


def cross_numeric_features(numeric_df: pd.DataFrame, n_crossings: int=2):
    numeric_df = numeric_df.copy()
    cross_features = []
    for cols in it.combinations(numeric_df, r=n_crossings):
        cross_feature = ",".join(cols)
        numeric_df[cross_feature] = 1.0
        for c in cols:
            numeric_df[cross_feature] *= numeric_df[c]
        cross_features.append(cross_feature)
    
    numeric_cross_df = numeric_df[cross_features]
    return numeric_cross_df
    

def cross_features(feature_df: pd.DataFrame, n_crossings: int=2):
    numeric_cols = get_numeric_cols(feature_df)
    string_cols = get_categorical_cols(feature_df)

    numeric_cross_df = cross_numeric_features(feature_df[numeric_cols], n_crossings)
    string_cross_df = cross_string_features(feature_df[string_cols], n_crossings)

    return pd.concat([string_cross_df, numeric_cross_df], axis=1)


def fit_target_encoders(feature_df: pd.DataFrame, target_df: pd.DataFrame):
    encoders = {}
    feature_df = feature_df.copy()
    df = pd.concat([feature_df, target_df], axis=1)
    impute_val = target_df.to_numpy().mean()
    cat_cols = get_categorical_cols(feature_df)

    for c in cat_cols:
        # Finish this implementation
        subset_cols = [c, "target"]
        encoding_df = df[subset_cols].set_index(c).groupby(c).mean()
        levels = list(encoding_df.index.to_flat_index())
        targets_per_level = encoding_df["target"].to_numpy()
        cat_to_label = dict(zip(levels, targets_per_level))
        encoders[c] = [cat_to_label, impute_val]
        
    return encoders


def fit_label_encoders(feature_df, n_crossings: int=1):
    df = feature_df.copy()
    encoders = {}
    categorical_cols = get_categorical_cols(df)
    for c in categorical_cols:
        categories = sorted(df[c].unique())
        cardinality = len(categories)
        labels = range(cardinality)
        cat_to_label = dict(zip(categories, labels))
        most_common_category = df[c].mode().to_numpy()[0]
        impute_val = cat_to_label[most_common_category]
        encoders[c] = [cat_to_label, impute_val]

    return encoders


def transform_categorical(df, encoders):
    df = df.copy()
    categorical_cols = get_categorical_cols(df)
    for c in categorical_cols:
        enc, missing_val = encoders[c]
        df[c] = df[c].apply(lambda x: enc.get(x, missing_val))
    return df


def fit_one_hot_encoder(df: pd.DataFrame):
    cat_cols = get_categorical_cols(df)
    df_cat = df[cat_cols]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False).fit(df_cat)
    
    return encoder


def transform_one_hot(df, one_hot_encoder):
    cont_cols = get_numeric_cols(df)
    categorical_cols = get_categorical_cols(df)
    
    df_cont = df[cont_cols]
    df_cat = df[categorical_cols]
    x_cat = one_hot_encoder.transform(df_cat)
    
    return np.concatenate([x_cat, df_cont.to_numpy()], axis=1)


def aggregate_numeric_features(feature_df: pd.DataFrame):
    categorical_cols = get_categorical_cols(feature_df)
    cont_cols = get_numeric_cols(feature_df)
    for cat_feat in categorical_cols:
        for cont_feat in cont_cols:
            agg_df = feature_df.groupby([cat_feat])[cont_feat].agg([np.mean, np.std]).reset_index()

    return None