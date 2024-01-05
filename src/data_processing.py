from typing import Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def categorical_to_int(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].astype("category").cat.codes
    return df


def load_tripadvisor(path: Optional[str] = None) -> pd.DataFrame:
    if path is None:
        path = "data/Travel_TripAdvisor_v2/Data_TripAdvisor_v2.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"Rating": "target", "UserID": "user", "ItemID": "item"})
    # users = df["user"].unique().tolist()
    # items = df["item"].unique().tolist()
    # df["user"] = df["user"].apply(lambda x: users.index(x))
    # df["item"] = df["item"].apply(lambda x: items.index(x))
    df = categorical_to_int(df, "user")
    df = categorical_to_int(df, "item")
    return df


def load_frappe(path: Optional[str] = None, do_binning: bool = False) -> pd.DataFrame:
    if path is None:
        path = "data/Mobile_Frappe/frappe/frappe.csv"
    df = pd.read_csv(path, sep="\t")
    df = categorical_to_int(df, "user")
    df = categorical_to_int(df, "item")
    if do_binning:
        df["bin_cnt"] = pd.qcut(df["cnt"], 11, labels=False, duplicates="drop")
        df = categorical_to_int(df, "bin_cnt")
        df = df.rename(columns={"bin_cnt": "target"})
    else:
        df = df.rename(columns={"cnt": "target"})
    return df


def check_nan(df: pd.DataFrame) -> None:
    # Check which columns has NaN values
    for column in df.columns:
        if df[column].isnull().any():
            print(f"{column} contains NaN values")


def convert_df_to_utility_mat(df: pd.DataFrame) -> np.array:
    user_list = df["user"].unique().tolist()
    item_list = df["item"].unique().tolist()
    num_users = len(user_list)
    num_items = len(item_list)

    utility_mat = np.full((num_items, num_users), np.nan)
    for _, row in df.iterrows():
        user_idx = user_list.index(row["user"])
        item_idx = item_list.index(row["item"])
        utility_mat[item_idx, user_idx] = row["target"]

    return utility_mat


def normalize_utility_mat(
    mat: np.array, user_based: bool = False
) -> Tuple[np.array, np.array]:
    if not user_based:
        means = np.nanmean(mat, axis=1)
        means = np.nan_to_num(means, nan=0.0)
        normalized_mat = mat - means.reshape(-1, 1)
        normalized_mat = np.nan_to_num(normalized_mat, nan=0.0)
    else:
        means = np.nanmean(mat, axis=0)
        means = np.nan_to_num(means, nan=0.0)
        normalized_mat = mat - means
        normalized_mat = np.nan_to_num(normalized_mat, nan=0.0)
    return normalized_mat, means


def split_data(
    df: pd.DataFrame,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Split data into train, valid, test
    # Ensuring that there is no unknown user or item in valid and test set
    subset_for_each_user = (
        df.groupby("user").apply(lambda x: x.sample(n=1)).droplevel("user")
    )
    subset_for_each_item = (
        df.groupby("item").apply(lambda x: x.sample(n=1)).droplevel("item")
    )
    subset = pd.concat(
        [subset_for_each_user, subset_for_each_item], axis=0
    ).drop_duplicates()
    remaining_data = df.drop(subset.index)

    df_train, df_eval = train_test_split(
        remaining_data, test_size=(valid_size + test_size), random_state=random_state
    )
    if valid_size != 0:
        df_valid, df_test = train_test_split(
            df_eval,
            test_size=test_size / (valid_size + test_size),
            random_state=random_state,
        )
    else:
        df_valid = None
        df_test = df_eval
    df_train = pd.concat([df_train, subset], axis=0)
    return df_train, df_valid, df_test


def split_data_cv(
    df: pd.DataFrame,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple:
    # Split data into train, valid, test
    # Ensuring that there is no unknown user or item in valid and test set
    subset_for_each_user = (
        df.groupby("user").apply(lambda x: x.sample(n=1)).droplevel("user")
    )
    subset_for_each_item = (
        df.groupby("item").apply(lambda x: x.sample(n=1)).droplevel("item")
    )
    subset = pd.concat(
        [subset_for_each_user, subset_for_each_item], axis=0
    ).drop_duplicates()
    remaining_data = df.drop(subset.index)
    size_to_split = len(df) / (len(remaining_data) * cv)
    # print(size_to_split, len(df), len(remaining_data), len(subset))

    indices_train = []
    indices_valid = []
    for i in range(cv):
        df_train, df_val = train_test_split(
            remaining_data, test_size=size_to_split, random_state=random_state + i
        )
        df_train = pd.concat([df_train, subset], axis=0)
        indices_train.append(df_train.index)
        indices_valid.append(df_val.index)
        # print(len(indices_train[-1]), len(indices_valid[-1]))
    return zip(indices_train, indices_valid)
