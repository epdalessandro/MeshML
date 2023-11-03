import pandas as pd
import numpy as np
from scipy.linalg import logm, expm
from scipy.stats import zscore

def read_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)

def clip(val, threshold, mean):
    if(val > (mean + threshold)): return mean + threshold
    elif(val < (mean - threshold)): return mean - threshold
    else: return val

def clip_dataframe(in_df: pd.DataFrame, n_sigma) -> pd.DataFrame:
    """
    in_df: the input dataframe
    n_sigma: the number of standard deviations we'll clip values to
    """
    df = in_df.copy()
    df["X_Pos"] = df.apply(lambda row: clip(row["X_Pos"], df["X_Pos"].std() * n_sigma, df["X_Pos"].mean()), axis=1)
    df["Y_Pos"] = df.apply(lambda row: clip(row["Y_Pos"], df["Y_Pos"].std() * n_sigma, df["Y_Pos"].mean()), axis=1)
    # df["Area"] = df.apply(lambda row: clip(row["Area"], df["Area"].std() * n_sigma, df["Area"].mean()), axis=1)
    df["Wall_Dist"] = df.apply(lambda row: clip(row["Wall_Dist"], df["Wall_Dist"].std() * n_sigma, df["Wall_Dist"].mean()), axis=1)
    df["G_Wall_Dist_X"] = df.apply(lambda row: clip(row["G_Wall_Dist_X"], df["G_Wall_Dist_X"].std() * n_sigma, df["G_Wall_Dist_X"].mean()), axis=1)
    df["G_Wall_Dist_Y"] = df.apply(lambda row: clip(row["G_Wall_Dist_Y"], df["G_Wall_Dist_Y"].std() * n_sigma, df["G_Wall_Dist_Y"].mean()), axis=1)
    df["AoA"] = df.apply(lambda row: clip(row["AoA"], df["AoA"].std() * n_sigma, df["AoA"].mean()), axis=1)
    df["M"] = df.apply(lambda row: clip(row["M"], df["M"].std() * n_sigma, df["M"].mean()), axis=1)
    df["Re"] = df.apply(lambda row: clip(row["Re"], df["Re"].std() * n_sigma, df["Re"].mean()), axis=1)
    return df

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.min()) / (df.max() - df.min())

def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std()

def metric_log(A, B, C):
    M = np.array([[A,B],[B,C]])
    log_metric = logm(M)
    return log_metric[0,0], log_metric[0,1], log_metric[1,1]

def metric_exp(A, B, C):
    M = np.array([[A,B],[B,C]])
    exp_metric = expm(M)
    return exp_metric[0,0], exp_metric[0,1], exp_metric[1,1]

def remove_outliers(in_df: pd.DataFrame, n_sigma: int) -> pd.DataFrame:
    # zscore gets the number of standard deviations away a value is
    # .all(axis=1) ensures that if a value in any column exceeds the threshold, we drop the row
    return in_df[(np.abs(zscore(in_df)) < n_sigma).all(axis=1)]

# Assumes a dataframe with X_Pos, Y_Pos, Area, Wall_Dist, G_Wall_Dist, Re, alpha 
# and Mach number as input with Matrix Metrics as outputs
def process_dataframe(in_df: pd.DataFrame) -> pd.DataFrame:
    # Normalize X_Pos, Y_Pos, Wall_Dist, G_Wall_Dist
    df = in_df.copy()
    df["X_Pos"] = df["X_Pos"] / df["X_Pos"].abs().max()
    df["Y_Pos"] = df["Y_Pos"] / df["Y_Pos"].abs().max()
    df["Wall_Dist"] = df["Wall_Dist"] / df["Wall_Dist"].abs().max()
    df["G_Wall_Dist_X"] = df["G_Wall_Dist_X"] / df["G_Wall_Dist_X"].abs().max()
    df["G_Wall_Dist_Y"] = df["G_Wall_Dist_Y"] / df["G_Wall_Dist_Y"].abs().max()

    # df["Area"] = df["Area"] / df["Area"].abs().max() # TODO: Should I normalize this by max value?
    df["AoA"] = df["AoA"] / df["AoA"].abs().max() # TODO: Should I normalize this by max value?
    df["M"] = df["M"] / df["M"].abs().max() # TODO: Should I normalize this by max value?
    df["Re"] = df["Re"] / df["Re"].abs().max() # TODO: Should I normalize this by max value?

    # df["A"], df["B"], df["C"] = zip(*df.apply(lambda row: metric_log(row["A"], row["B"], row["C"]), axis = 1))

    # Don't normalize Re, alpha or Mach Number
    return df

def norm_process(in_df: pd.DataFrame) -> pd.DataFrame:
    df = in_df.copy()
    # Change mesh metrics to metric log
    df["A"], df["B"], df["C"] = zip(*df.apply(lambda row: metric_log(row["A"], row["B"], row["C"]), axis = 1))
    df = (df - df.mean()) / df.std() # Normalize distribution of each column to mean 0 variance 1
    return df

def get_split(df: pd.DataFrame, train_split: float=0.8, val_split: float=0.1, 
                test_split: float=0.1) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    assert((train_split + val_split + test_split) == 1)
    # sample = df.sample(frac=1, random_state=0) //TODO: Check if I need this
    split_indices = [int(len(df)*train_split), int((1-test_split) * len(df))]
    train, val, test = np.split(df, split_indices, axis=0)
    return train.to_numpy(), val.to_numpy(), test.to_numpy()

def main():
    # TODO: 5 standard deviations for clipping is an arbitrary value, figure out something else
    df = read_data("../data/aggregate_data.csv")

    clipped_df = clip_dataframe(df,1)
    clipped_df.to_csv("../data/clipped_data.csv", index=False)

    no_outliers = remove_outliers(df, 3) # For a normal distribution >3 standard deviations is an outlier
    no_outliers.to_csv("../data/no_outliers_data.csv", index=False)

    processed_df_clipped = process_dataframe(clipped_df)
    processed_df_clipped.to_csv("../data/processed_clipped_data.csv", index=False)

    processed_df_outliers = process_dataframe(no_outliers)
    processed_df_outliers.to_csv("../data/processed_outlier_data.csv", index=False)

    norm_clip_df = norm_process(df)
    norm_clip_df.to_csv("../data/norm_clipped_data.csv", index=False)

    norm_outlier_df = norm_process(df)
    norm_outlier_df.to_csv("../data/norm_outlier_data.csv", index=False)

if __name__ == "__main__":
    main()