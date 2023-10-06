import pandas as pd
import numpy as np
from scipy.linalg import logm
from scipy.stats import zscore
import matplotlib.pyplot as plt

# df = pd.read_csv("./data/aggregate_data.csv")
# print("\n\nAggregate")
# print(df.describe())
# dfHist = df.hist()

# clipped = pd.read_csv("./data/clipped_data.csv")
# print("\n\nClipped")
# print(clipped.describe())
# clippedHist = clipped.hist()

processed = pd.read_csv("../data/processed_data.csv")
print("\n\nProcessed")
print(processed.describe())
processedHist = processed.hist()

# norm = pd.read_csv("./data/norm_data.csv")
# print("\n\nNorm")
# print(norm.describe())
# normHist = norm.hist()
# plt.show()


# data = {
#             'A':[1, 1,2,2,3,1,12, 20],
#             'B':[4, 4,4,4,5,5,5, 6],
#             'C':[7, 7,7,37,8,8,8, 9] }

# def remove_outliers(in_df: pd.DataFrame, n_sigma: int) -> pd.DataFrame:
#     return in_df[(np.abs(zscore(in_df)) < n_sigma).all(axis=1)]

# def add(a, b, c):
#     # return [c, a+b, a+b+c]
#     return [1,2,3]
# # Convert the dictionary into DataFrame
# df = pd.DataFrame(data)
# print(df.describe())
# df = remove_outliers(df, 1)
# print(df.describe())
# print("Original DataFrame:\n", df)

# def clip(val, threshold, mean):
#     if(val > mean+threshold): return mean+threshold
#     elif(val < mean-threshold): return mean-threshold
#     else: return val

# df["clipped"] = df.apply(lambda row: clip(row["A"], df["A"].std()*0.75, df["A"].mean()), axis=1)
# print(df["A"].std())
# print(df["B"].std())
# print(df["C"].std())
# dfHist = df.plot.hist()
# # plt.show()

# print('\nAfter Applying Function: ')
# # printing the new dataframe
# print(df)
# print(df["A"].mean())

# def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
#     normalized = (df-df.min())/(df.max() - df.min())
#     return normalized

# df = pd.DataFrame([1,2,3,4,5,6,7,8,9])
# print(df.describe())
# df_norm = normalize_dataframe(df)
# print(df.describe())
# print(df_norm.describe())


# df = pd.read_csv("./data/aggregate_data.csv")
# print(df["Area"].describe())
# print(df["Area"])

# df['add'], df['add2'], df['add3'] = zip(*df.apply(lambda row : add(row['A'], row['B'], row['C']), axis = 1))
# df['add'], df['add2'], df['add3'] = j
# df['one', 'two', 'three'] = df.apply(lambda row: add(row['A'], row['B'], row['C']), axis = 1)
# def metric_log(A, B, C):
#     M = np.array([[A,B],[B,C]])
#     log_metric = logm(M)
#     return log_metric[0,0], log_metric[0,1], log_metric[1,0]

# df["A"], df["B"], df["C"] = zip(*df.apply(lambda row: metric_log(row["A"], row["B"], row["C"]), axis = 1))
