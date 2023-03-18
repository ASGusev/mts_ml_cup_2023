from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import seq_solutions
import feature_utils


LIGHTNING_ROOT = Path('lightning_logs')
LIGHTNING_PARAMS_FILENAME = 'hparams.yaml'
LIGHTNING_CHECKPOINT_DIR_NAME = 'checkpoints'


def to_cuda(obj):
    if isinstance(obj, dict):
        return {fn: to_cuda(fv) for fn, fv in obj.items()}
    elif isinstance(obj, list):
        return [to_cuda(i) for i in obj]
    return obj.cuda()


def main(args):
    lightning_run_dir = LIGHTNING_ROOT / f'version_{args.lightning_version}'
    checkpoint_dir = lightning_run_dir / LIGHTNING_CHECKPOINT_DIR_NAME
    checkpoint_path, = checkpoint_dir.iterdir()
    model = seq_solutions.ConvModel.load_from_checkpoint(checkpoint_path).to('cuda')
    model.eval()
    conf = OmegaConf.load(args.config_path)

    train_data = pd.read_parquet(feature_utils.DATA_ROOT / 'train_users.pqt')
    val_data = pd.read_parquet(feature_utils.DATA_ROOT / 'val_users.pqt')
    train_data.loc[train_data['is_male'] == 'NA', 'is_male'] = 'nan'
    val_data.loc[val_data['is_male'] == 'NA', 'is_male'] = 'nan'

    user_feature_loader = feature_utils.FeatureLoader(conf['features']['user'], feature_utils.USER_FEATURES_DIR)
    url_feature_loader = feature_utils.FeatureLoader(conf['features']['url'], feature_utils.URL_FEATURES_DIR)
    url_features = url_feature_loader.get_num_features(np.arange(feature_utils.N_URLS))
    urls_embeddings = [
        feature_utils.load_urls_embeddings(feature_utils.EMBEDDINGS_DIR / en)
        for en in conf['features']['url_embeddings_names']
    ]
    user_embeddings = feature_utils.PartialEmbeddingsLoader(
        np.concatenate((train_data['user_id'].values, val_data['user_id'].values)),
        [feature_utils.EMBEDDINGS_DIR / en / 'users.npy' for en in conf['features']['user_embeddings_names']]
    )

    interaction_sets = np.load('interactions/interactions.npy', allow_pickle=True)
    interaction_features = feature_utils.load_interaction_features(conf['features'])

    users_data = pd.read_parquet(args.data_path)
    user_ids = users_data['user_id'].values
    target = np.full_like(user_ids, -1)
    if conf['target'] == 'is_male':
        target = target.astype(np.float32).reshape((-1, 1))
    else:
        target = feature_utils.age_bucket(target).astype(np.int64)
    dataset = seq_solutions.UsersDataset(
        user_ids, target,
        interaction_sets[user_ids], interaction_features[user_ids],
        user_feature_loader.get_cat_features(user_ids), user_feature_loader.get_num_features(user_ids),
        user_embeddings[user_ids],
        urls_embeddings, url_features,
        conf['batch_size'], conf['interactions_limit'], order_by_len=False
    )
    with torch.no_grad():
        predictions = [
            model(to_cuda(batch)).cpu().numpy()
            for batch in tqdm(dataset)
        ]
    predictions = np.concatenate(predictions)
    np.save(f'{args.out_dir}/seq_{args.lightning_version}.npy', predictions)


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('config_path', type=Path)
    argument_parser.add_argument('lightning_version', type=int)
    argument_parser.add_argument('data_path', type=Path)
    argument_parser.add_argument('target')
    argument_parser.add_argument('out_dir', type=Path)
    arguments = argument_parser.parse_args()

    main(arguments)
