import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config, data_loading

# LoL experimental scenarios
SCENARIO_DRAFT_ONLY = "draft_only"        # Scenario A: pre-match information only
SCENARIO_DRAFT_INGAME = "draft_ingame"    # Scenario B: draft + in-game statistics
LOL_SCENARIOS = (SCENARIO_DRAFT_ONLY, SCENARIO_DRAFT_INGAME)


def _scatter_add(matrix, ids, vocabulary, value):
    positions = np.searchsorted(vocabulary, ids)
    positions = np.minimum(positions, len(vocabulary) - 1)
    valid = vocabulary[positions] == ids
    rows = np.nonzero(valid)[0]
    matrix[rows, positions[valid]] += value


def _unique_positive_ids(df, columns):
    values = np.unique(np.concatenate([df[c].to_numpy() for c in columns]))
    return values[values > 0]


def _signed_team_encoding(values):
    values = np.asarray(values)
    return np.where(values == 1, 1.0, np.where(values == 2, -1.0, 0.0))



class ChampionPresenceEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, pick_cols, ban_cols, spell_cols=None):
        self.pick_cols = pick_cols
        self.ban_cols = ban_cols
        self.spell_cols = spell_cols

    def fit(self, X, y=None):
        all_picks = [c for cols in self.pick_cols.values() for c in cols]
        all_bans = [c for cols in self.ban_cols.values() for c in cols]
        self.pick_vocab_ = _unique_positive_ids(X, all_picks)
        self.ban_vocab_ = _unique_positive_ids(X, all_bans)

        if self.spell_cols:
            all_spells = [c for cols in self.spell_cols.values() for c in cols]
            self.spell_vocab_ = _unique_positive_ids(X, all_spells)
        else:
            self.spell_vocab_ = np.array([], dtype=int)
        return self

    def transform(self, X):
        n = len(X)
        picks = np.zeros((n, len(self.pick_vocab_)), dtype=np.float32)
        bans = np.zeros((n, len(self.ban_vocab_)), dtype=np.float32)

        for team, sign in ((1, 1.0), (2, -1.0)):
            for col in self.pick_cols[team]:
                _scatter_add(picks, X[col].to_numpy(), self.pick_vocab_, sign)
            for col in self.ban_cols[team]:
                _scatter_add(bans, X[col].to_numpy(), self.ban_vocab_, sign)

        blocks = [picks, bans]

        if self.spell_cols:
            for team in (1, 2):
                spells = np.zeros((n, len(self.spell_vocab_)), dtype=np.float32)
                for col in self.spell_cols[team]:
                    _scatter_add(spells, X[col].to_numpy(), self.spell_vocab_, 1.0)
                blocks.append(spells)

        return np.hstack(blocks)

    def get_feature_names_out(self, input_features=None):
        names = [f"pick_{i}" for i in self.pick_vocab_]
        names += [f"ban_{i}" for i in self.ban_vocab_]
        if self.spell_cols:
            for team in (1, 2):
                names += [f"t{team}_spell_{i}" for i in self.spell_vocab_]
        return np.array(names, dtype=object)


def prepare_dota2():
    df_train, df_test = data_loading.load_dota2()

    y_train = df_train[config.TARGET_COL]
    y_test = df_test[config.TARGET_COL]
    X_train = df_train.drop(columns=[config.TARGET_COL])
    X_test = df_test.drop(columns=[config.TARGET_COL])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.DOTA2_CATEGORICAL_COLS,
            ),
            ("heroes", "passthrough", config.DOTA2_HERO_COLS),
        ],
        verbose_feature_names_out=False,
    )

    X_train_t = preprocessor.fit_transform(X_train)   # fit on TRAIN only
    X_test_t = preprocessor.transform(X_test)
    names = preprocessor.get_feature_names_out()

    X_train = pd.DataFrame(X_train_t.astype(np.float32), columns=names, index=y_train.index)
    X_test = pd.DataFrame(X_test_t.astype(np.float32), columns=names, index=y_test.index)
    return X_train, X_test, y_train, y_test


def _build_lol_preprocessor(scenario):
    draft_cols = (
        [c for cols in config.LOL_PICK_COLS.values() for c in cols]
        + [c for cols in config.LOL_BAN_COLS.values() for c in cols]
        + [c for cols in config.LOL_SPELL_COLS.values() for c in cols]
    )

    transformers = [
        (
            "draft",
            ChampionPresenceEncoder(
                pick_cols=config.LOL_PICK_COLS,
                ban_cols=config.LOL_BAN_COLS,
                spell_cols=config.LOL_SPELL_COLS,
            ),
            draft_cols,
        )
    ]

    if scenario == SCENARIO_DRAFT_INGAME:
        transformers += [
            (
                "first_events",
                FunctionTransformer(
                    _signed_team_encoding, feature_names_out="one-to-one"
                ),
                config.LOL_FIRST_EVENT_COLS,
            ),
            ("ingame_numeric", StandardScaler(), config.LOL_INGAME_NUMERIC_COLS),
        ]

    # remainder="drop" discards creationTime, seasonId and anything unused.
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_lol(scenario=SCENARIO_DRAFT_ONLY):
    if scenario not in LOL_SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Use one of {LOL_SCENARIOS}.")

    df = data_loading.load_lol()
    y = df[config.TARGET_COL]
    X = df.drop(columns=[config.TARGET_COL] + config.LOL_DROP_COLS)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )

    preprocessor = _build_lol_preprocessor(scenario)
    X_train_t = preprocessor.fit_transform(X_train)   # fit on TRAIN only
    X_test_t = preprocessor.transform(X_test)
    names = preprocessor.get_feature_names_out()

    X_train = pd.DataFrame(X_train_t.astype(np.float32), columns=names, index=y_train.index)
    X_test = pd.DataFrame(X_test_t.astype(np.float32), columns=names, index=y_test.index)
    return X_train, X_test, y_train, y_test


def make_validation_split(X_train, y_train):
    
    return train_test_split(
        X_train,
        y_train,
        test_size=config.VAL_SIZE,
        stratify=y_train,
        random_state=config.RANDOM_STATE,
    )


if __name__ == "__main__":
    def report(name, X_train, X_test, y_train, y_test):
        print(f"\n=== {name} ===")
        print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
        print(f"y_train mean: {y_train.mean():.4f} | y_test mean: {y_test.mean():.4f}")
        print(f"NaNs in X_train: {int(X_train.isna().sum().sum())}")

    report("Dota 2", *prepare_dota2())
    for sc in LOL_SCENARIOS:
        report(f"LoL - {sc}", *prepare_lol(sc))