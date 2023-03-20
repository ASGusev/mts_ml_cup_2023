from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import feature_utils
import user_features_models as ufm


def main(args):
    model = ufm.load_from_run_dir(args.run_dir)
    conf = OmegaConf.load(args.run_dir / '.hydra' / 'config.yaml')

    data_file_name = 'val_users.pqt' if args.dataset == 'val' else 'submit_2.pqt'
    users_data = pd.read_parquet(feature_utils.DATA_ROOT / data_file_name)
    user_ids = users_data['user_id'].values

    user_feature_loader = feature_utils.FeatureLoader(conf['features'], feature_utils.USER_FEATURES_DIR)
    embedding_files = conf['features']['embeddings_names']
    if embedding_files:
        user_embeddings = feature_utils.PartialEmbeddingsLoader(
            user_ids,
            [feature_utils.EMBEDDINGS_DIR / en / 'users.npy' for en in embedding_files]
        )
    else:
        user_embeddings = None

    target = conf['target']

    dataset = ufm.DS(user_feature_loader, user_embeddings, target, user_ids)
    prediction = model.predict_probability(dataset)
    out_filename = f'{conf["model_class"]}_{"_".join(args.run_dir.parts[-2:])}.npy'
    out_path = f'raw_predictions/{args.dataset}_{conf["target"]}/{out_filename}'
    np.save(out_path, prediction)


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('run_dir', type=Path)
    argument_parser.add_argument('dataset', choices=('val', 'test'))
    arguments = argument_parser.parse_args()

    main(arguments)
