import tensorflow as tf
import pandas as pd
from pathlib import Path

def load_ml_1m():
    # download and extract zip file
    tf.keras.utils.get_file(
        "ml-1m.zip",
        "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        cache_dir=".",
        cache_subdir=".",
        extract=True,
    )
    # read and merge data into same table
    cur_path = Path(".").absolute()
    ratings = pd.read_csv(
        cur_path / "ml-1m" / "ratings.dat",
        sep="::",
        usecols=[0, 1, 2, 3],
        names=["user", "item", "rating", "time"],
    )
    users = pd.read_csv(
        cur_path / "ml-1m" / "users.dat",
        sep="::",
        usecols=[0, 1, 2, 3],
        names=["user", "sex", "age", "occupation"],
    )
    items = pd.read_csv(
        cur_path / "ml-1m" / "movies.dat",
        sep="::",
        usecols=[0, 1, 2],
        names=["item", "title", "genre"],
        encoding="iso-8859-1",
    )
    items[["genre1", "genre2", "genre3"]] = (
        items["genre"].str.split(r"|", expand=True).fillna("missing").iloc[:, :3]
    )
    items.drop("genre", axis=1, inplace=True)
    data = ratings.merge(users, on="user").merge(items, on="item")
    data.rename(columns={"rating": "label"}, inplace=True)
    # random shuffle data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    return data