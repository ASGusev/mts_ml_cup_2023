from pathlib import Path
import bisect

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import hydra
import omegaconf

import feature_utils
import seq_solutions

DATA_ROOT = Path('data/')
CONVERTED_DATA_ROOT = DATA_ROOT / 'data_converted'
USER_FEATURES_DIR = Path('user_features/')
URL_FEATURES_DIR = Path('url_features/')
FEATURE_TRANSFORMERS_DIR = Path('feature_transformers/')
EMBEDDINGS_DIR = Path('embeddings/')
INTERACTIONS_DIR = Path('interactions')
N_URLS = 199683
N_USERS = 415317


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

        self.mean_calculators = [
            feature_utils.KeyedMeanCalculator.load(features_dir / f'{fn}.npz')
            for fn in conf['mean_calculators']
        ]

    def get_cat_features(self, indexes):
        return np.stack([
            self.cat_tops[fn][indexes]
            for fn in self.cat_feature_names
        ]).T

    def get_num_features(self, indexes):
        return np.stack((
            *(self.cat_numbs[fn][indexes] for fn in self.cat_feature_names),
            *(mc.get(indexes) for mc in self.mean_calculators),
            *self.ready_features[:, indexes]
        )).T


def load_interaction_features(conf):
    feature_arrays = load_feature_arrays(conf['interaction_features'], INTERACTIONS_DIR).T
    return np.array([np.stack(ufs, axis=1) for ufs in feature_arrays], dtype=object)


@hydra.main(version_base=None, config_path='.', config_name='run_conf.yaml')
def main(conf: omegaconf.DictConfig):
    train_data = pd.read_parquet(DATA_ROOT / 'train_users.pqt')
    val_data = pd.read_parquet(DATA_ROOT / 'val_users.pqt')
    train_data.loc[train_data['is_male'] == 'NA', 'is_male'] = 'nan'
    val_data.loc[val_data['is_male'] == 'NA', 'is_male'] = 'nan'

    user_feature_loader = FeatureLoader(conf['features']['user'], USER_FEATURES_DIR)
    url_feature_loader = FeatureLoader(conf['features']['url'], URL_FEATURES_DIR)
    url_features = url_feature_loader.get_num_features(np.arange(N_URLS))
    urls_embeddings = [
        feature_utils.load_urls_embeddings(EMBEDDINGS_DIR / en)
        for en in conf['features']['url_embeddings_names']
    ]
    user_embeddings = feature_utils.PartialEmbeddingsLoader(
        np.concatenate((train_data['user_id'].values, val_data['user_id'].values)),
        [EMBEDDINGS_DIR / en / 'users.npy' for en in conf['features']['user_embeddings_names']]
    )

    interaction_sets = np.load('interactions/interactions.npy', allow_pickle=True)
    interaction_features = load_interaction_features(conf['features'])

    def make_ds(data):
        user_ids = data['user_id'].values
        target = data[conf['target']].values
        if conf['target'] == 'is_male':
            target = target.astype(np.float32).reshape((-1, 1))
        else:
            target = age_bucket(target).astype(np.int64)
        return seq_solutions.UsersDataset(
            user_ids, target,
            interaction_sets[user_ids], interaction_features[user_ids],
            user_feature_loader.get_cat_features(user_ids), user_feature_loader.get_num_features(user_ids),
            user_embeddings[user_ids],
            urls_embeddings, url_features,
            conf['batch_size'], conf['interactions_limit']
        )

    torch.manual_seed(42)
    np.random.seed(42)
    train_ds = make_ds(train_data)
    train_ds = seq_solutions.DatasetShuffleWrapper(train_ds)
    val_ds = make_ds(val_data)

    out_dim = 1 if conf['target'] == 'is_male' else 7
    model = seq_solutions.ConvModel(**conf['model_parameters'], out_dim=out_dim)
    trainer = pl.Trainer(**conf['trainer_parameters'], callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=16)])
    # noinspection PyTypeChecker
    trainer.fit(model, train_ds, val_ds)


if __name__ == '__main__':
    main()
