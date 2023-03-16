import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm


class CatNumerator:
    def __init__(self, cats=None):
        self.cats = cats if cats else []
        self.cat_ids = {c: i for i, c in enumerate(self.cats)}

    def update(self, new_cats):
        new_cats = [c for c in new_cats if c not in self.cats]
        for i, c in enumerate(new_cats):
            self.cat_ids[c] = i + len(self.cats)
        self.cats.extend(new_cats)

    def transform(self, x):
        if isinstance(x, pd.Series):
            x = x.values
        if np.isscalar(x):
            return self.cat_ids[x]
        elif isinstance(x, np.ndarray):
            return np.array([self.cat_ids[i] for i in x])
        raise ValueError
    
    def inv_transform(self, x):
        if isinstance(x, pd.Series):
            x = x.values
        if np.isscalar(x):
            return self.cats[x]
        elif isinstance(x, np.ndarray):
            return np.array([self.cats[i] for i in x])
        raise ValueError

    def __len__(self):
        return len(self.cats)

    def save(self, path):
        with open(path, 'wt') as f:
            json.dump(self.cats, f)

    @staticmethod
    def load(path):
        with open(path) as f:
            cats = json.load(f)
        return CatNumerator(cats)


class ZeroOneScaler:
    def __init__(self, min_=float('inf'), max_=-float('inf')):
        self.min = min_
        self.max = max_

    def update(self, x):
        self.min = min(self.min, min(x))
        self.max = max(self.max, max(x))

    def transform(self, x):
        return (x - self.min) / (self.max - self.min)
    
    def inv_transform(self, x):
        return x * (self.max - self.min) + self.min

    def save(self, path):
        with open(path, 'wt') as f:
            json.dump(
                {
                    'min': self.min,
                    'max': self.max
                },
                f
            )

    @staticmethod
    def load(path):
        with open(path) as f:
            state = json.load(f)
        return ZeroOneScaler(
            min_=state['min'], 
            max_=state['max']
        )


def read_dir(path):
    for part_path in tqdm(list(path.glob('*.parquet'))):
        yield pd.read_parquet(part_path)


def load_users_embeddings(directory: Path):
    return np.load(str(directory / 'users.npy'))


def load_urls_embeddings(directory: Path):
    embeddings = np.load(str(directory / 'urls.npy'))
    mapping_path = directory / 'url_mapping.npy'
    if mapping_path.exists():
        mapping = np.load(str(mapping_path))
        return embeddings[mapping]
    return embeddings


class KeyedMeanCalculator:
    def __init__(self, n=-1, *, sums=None, counters=None):
        self.sums = sums if sums is not None else np.zeros(n)
        self.counters = counters if counters is not None else np.zeros(n, np.int32)
    
    def get_global(self):
        return self.sums.sum() / self.counters.sum()

    def get(self, keys, *, vals_to_exclude=None, multipliers_to_exclude=1, default=None):
        sums = self.sums[keys]
        counters = self.counters[keys]
        if vals_to_exclude is not None:
            sums -= vals_to_exclude * multipliers_to_exclude
            counters -= multipliers_to_exclude
        unseen_mask = counters == 0
        if unseen_mask.sum():
            if default is None:
                default = self.get_global()
            sums[unseen_mask] = default
            counters[unseen_mask] = 1
        return sums / counters
    
    def update(self, keys, new_vals, multipliers=1):
        np.add.at(self.sums, keys, new_vals * multipliers)
        np.add.at(self.counters, keys, multipliers)

    def save(self, path):
        np.savez(path, sums=self.sums, counters=self.counters)

    @staticmethod
    def load(path):
        state = np.load(path)
        return KeyedMeanCalculator(**state)


class CatStatSummarizer:
    def __init__(self, n_keys, n_cats):
        self.shape = n_keys, n_cats
        self.counters = csr_matrix(self.shape, dtype=np.int32)

    def update(self, keys, cats, values):
        counters_update = csr_matrix((values, (keys, cats)), self.shape, dtype=np.int32)
        self.counters += counters_update

    def get_top_cats(self):
        return self.counters.argmax(axis=1).A1

    def get_cat_numbs(self):
        return (self.counters > 0).sum(axis=1).A1


class PartialEmbeddingsLoader:
    def __init__(self, indexes, embeddings_paths):
        indexes = np.sort(indexes)
        # noinspection PyArgumentList
        self.index_mapping = np.full(indexes.max() + 1, -1, np.int32)
        self.index_mapping[indexes] = np.arange(len(indexes))
        self.embeddings = [
            np.load(p)[indexes].copy()
            for p in embeddings_paths
        ]

    def __getitem__(self, request_indexes):
        remapped_indexes = self.index_mapping[request_indexes]
        return np.concatenate(
            [
                e[remapped_indexes]
                for e in self.embeddings
            ],
            axis=1
        )
