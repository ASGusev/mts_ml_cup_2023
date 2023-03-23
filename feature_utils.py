import bisect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm


DATA_ROOT = Path('data/')
CONVERTED_DATA_ROOT = DATA_ROOT / 'data_converted'
USER_FEATURES_DIR = Path('user_features/')
URL_FEATURES_DIR = Path('url_features/')
FEATURE_TRANSFORMERS_DIR = Path('feature_transformers/')
EMBEDDINGS_DIR = Path('embeddings/')
INTERACTIONS_DIR = Path('interactions')
N_URLS = 199683
N_USERS = 415317
N_AGE_BUCKETS = 7


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


@np.vectorize
def age_bucket(x):
    return bisect.bisect_left([18, 25, 35, 45, 55, 65], x)


def load_feature_arrays(conf, features_dir):
    return np.stack([
        np.load(str(features_dir / f'{fn}.npy'), allow_pickle=True)
        for fn in conf
    ])


class FeatureLoader:
    def __init__(self, conf, features_dir):
        self.cat_tops = {}
        self.cat_numbs = {}
        for fn in conf['cat_features']:
            stats = np.load(str(features_dir / f'{fn}.npz'))
            self.cat_tops[fn] = stats['top']
            self.cat_numbs[fn] = stats['numb']
        self.cat_feature_names = [f'{fn}' for fn in conf['cat_features']]

        self.ready_features = load_feature_arrays(conf['ready_features'], features_dir)
        self.ready_cat_features = load_feature_arrays(conf['ready_cat_features'], features_dir) \
            if conf['ready_cat_features'] else np.ndarray((0, N_USERS))

        self.mean_calculators = [
            KeyedMeanCalculator.load(features_dir / f'{fn}.npz')
            for fn in conf['mean_calculators']
        ]
        self.n_cat_features = len(self.cat_feature_names) + len(self.ready_cat_features)

    def get_cat_features(self, indexes):
        if not self.cat_feature_names and not self.ready_cat_features:
            return np.ndarray((len(indexes), 0))
        return np.stack((
            *(
                self.cat_tops[fn][indexes]
                for fn in self.cat_feature_names
            ),
            *self.ready_cat_features[:, indexes]
        )).T

    def get_num_features(self, indexes):
        return np.stack((
            *(self.cat_numbs[fn][indexes] for fn in self.cat_feature_names),
            *(mc.get(indexes) for mc in self.mean_calculators),
            *self.ready_features[:, indexes]
        )).T


def load_interaction_features(conf):
    feature_arrays = load_feature_arrays(conf['interaction_features'], INTERACTIONS_DIR).T
    return np.array([np.stack(ufs, axis=1) for ufs in feature_arrays], dtype=object)
