import numpy as np
import pandas as pd

from src import config


def _check_file_exists(path):
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Download the dataset following the instructions in README.md."
        )


def load_dota2():
    _check_file_exists(config.DOTA2_TRAIN_PATH)
    _check_file_exists(config.DOTA2_TEST_PATH)

    dataframes = []

    for path in (config.DOTA2_TRAIN_PATH, config.DOTA2_TEST_PATH):
        df = pd.read_csv(
            path, header=None, names=config.DOTA2_COLUMNS, dtype=np.int16
        )

        values = set(df["winner"].unique())
        if not values.issubset({-1, 1}):
            raise ValueError(
                f"Unexpected values in the Dota 2 target: {values}"
            )
        
        target = (df["winner"] == 1).astype(int).rename(config.TARGET_COL)
        df = pd.concat([df.drop(columns=["winner"]), target], axis=1)
        dataframes.append(df)

    return dataframes[0], dataframes[1]


def load_lol():
    _check_file_exists(config.LOL_GAMES_PATH)

    df = pd.read_csv(config.LOL_GAMES_PATH)

    if config.LOL_TARGET_SOURCE_COL not in df.columns:
        raise ValueError(
            f"Column '{config.LOL_TARGET_SOURCE_COL}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    
    values = set(df[config.LOL_TARGET_SOURCE_COL].unique())
    
    if not values.issubset({1, 2}):
        raise ValueError(
            f"Unexpected values in the LoL target: {values}"
        )
    
    df[config.TARGET_COL] = (df[config.LOL_TARGET_SOURCE_COL] == 1).astype(int)
    df = df.drop(columns=[config.LOL_TARGET_SOURCE_COL])

    if config.LOL_ID_COL in df.columns:
        df = df.drop(columns=[config.LOL_ID_COL])

    return df


def summarize_dataset(df, name):
    print(f"\n=== {name} ===")
    print(f"Instances: {df.shape[0]} | Features (excluding target): {df.shape[1] - 1}")

    balance = df[config.TARGET_COL].value_counts(normalize=True).sort_index()
    print("Target balance:")
    for label, proportion in balance.items():
        print(f"  class {label}: {proportion:.2%}")

    n_missing = int(df.isna().sum().sum())
    print(f"Missing values (total): {n_missing}")


if __name__ == "__main__":
    dota_train, dota_test = load_dota2()
    summarize_dataset(dota_train, "Dota 2 - train")
    summarize_dataset(dota_test, "Dota 2 - test")

    lol = load_lol()
    summarize_dataset(lol, "League of Legends")
    print("\nLoL columns:")
    print(list(lol.columns))