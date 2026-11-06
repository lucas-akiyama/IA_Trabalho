from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

DOTA2_DIR = DATA_RAW_DIR / "dota2"
DOTA2_TRAIN_PATH = DOTA2_DIR / "dota2Train.csv"
DOTA2_TEST_PATH = DOTA2_DIR / "dota2Test.csv"

LOL_DIR = DATA_RAW_DIR / "lol"
LOL_GAMES_PATH = LOL_DIR / "games.csv"

RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

RANDOM_STATE = 42

TEST_SIZE = 0.20

VAL_SIZE = 0.20

CV_FOLDS = 5

DOTA2_N_HEROES = 113
DOTA2_META_COLS = ["winner", "cluster_id", "game_mode", "game_type"]
DOTA2_HERO_COLS = [f"hero_{i}" for i in range(1, DOTA2_N_HEROES + 1)]
DOTA2_COLUMNS = DOTA2_META_COLS + DOTA2_HERO_COLS

DOTA2_CATEGORICAL_COLS = ["cluster_id", "game_mode", "game_type"]

LOL_ID_COL = "gameId"
LOL_TARGET_SOURCE_COL = "winner"

TARGET_COL = "target"

MLP_PARAM_GRID = {
    "hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64), (128, 64, 32)],
    "learning_rate_init": [0.0001, 0.001, 0.01],
    "alpha": [0.0001, 0.001],
}
MLP_FIXED_PARAMS = {
    "activation": "relu",
    "solver": "adam",
    "max_iter": 300,
    "early_stopping": True,
    "n_iter_no_change": 10,
    "random_state": RANDOM_STATE,
}

RF_PARAM_GRID = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [5, 10, 20, None],
}
RF_FIXED_PARAMS = {
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

LOL_N_PLAYERS = 5
LOL_TEAMS = (1, 2)

LOL_PICK_COLS = {
    t: [f"t{t}_champ{i}id" for i in range(1, LOL_N_PLAYERS + 1)] for t in LOL_TEAMS
}
LOL_BAN_COLS = {
    t: [f"t{t}_ban{i}" for i in range(1, LOL_N_PLAYERS + 1)] for t in LOL_TEAMS
}
LOL_SPELL_COLS = {
    t: [
        f"t{t}_champ{i}_sum{j}"
        for i in range(1, LOL_N_PLAYERS + 1)
        for j in (1, 2)
    ]
    for t in LOL_TEAMS
}

LOL_FIRST_EVENT_COLS = [
    "firstBlood", "firstTower", "firstInhibitor",
    "firstBaron", "firstDragon", "firstRiftHerald",
]
LOL_INGAME_NUMERIC_COLS = ["gameDuration"] + [
    f"t{t}_{stat}"
    for t in LOL_TEAMS
    for stat in (
        "towerKills", "inhibitorKills", "baronKills",
        "dragonKills", "riftHeraldKills",
    )
]

LOL_DROP_COLS = ["creationTime", "seasonId"]

# é possível alterar para "accuracy", "precision", "recall", "f1"
SELECTION_METRIC = "accuracy"  

SEEDS = [42, 43, 44, 45, 46]

# Reduced settings for a fast end-to-end check: python -m src.main --quick
QUICK_SEEDS = [42, 43]
QUICK_MLP_PARAM_GRID = {
    "hidden_layer_sizes": [(32,), (64, 32)],
    "learning_rate_init": [0.001, 0.01],
    "alpha": [0.0001],
}
QUICK_RF_PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10],
}

CLASS_NAMES = {
    "dota2": ("Dire won", "Radiant won"),
    "lol_draft_only": ("Team 2 won", "Team 1 won"),
    "lol_draft_ingame": ("Team 2 won", "Team 1 won"),
}
