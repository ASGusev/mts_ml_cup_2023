from pathlib import Path
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import omegaconf
import pandas as pd

import feature_utils
import user_features_models as ufm


def conf_to_dict(conf):
    if isinstance(conf, omegaconf.DictConfig):
        return {k: conf_to_dict(v) for k, v in conf.items()}
    return conf


class StdoutDuplicatedFileWriter:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.stdout = sys.stdout

    def write(self, data):
        with self.filepath.open('a') as f:
            f.write(data)
        self.stdout.write(data)

    def flush(self):
        pass


@hydra.main(version_base=None, config_path='user_features_conf', config_name='config.yaml')
def main(conf: omegaconf.DictConfig):
    # noinspection PyUnresolvedReferences
    run_dir = Path(HydraConfig.get()['run']['dir'])
    sys.stdout = StdoutDuplicatedFileWriter(run_dir / 'stdout_log.txt')
    train_data = pd.read_parquet(feature_utils.DATA_ROOT / 'train_users.pqt')
    val_data = pd.read_parquet(feature_utils.DATA_ROOT / 'val_users.pqt')
    train_data.loc[train_data['is_male'] == 'NA', 'is_male'] = 'nan'
    val_data.loc[val_data['is_male'] == 'NA', 'is_male'] = 'nan'

    user_feature_loader = feature_utils.FeatureLoader(conf['features'], feature_utils.USER_FEATURES_DIR)
    embedding_files = conf['features']['embeddings_names']
    if embedding_files:
        user_embeddings = feature_utils.PartialEmbeddingsLoader(
            np.concatenate((train_data['user_id'].values, val_data['user_id'].values)),
            [feature_utils.EMBEDDINGS_DIR / en / 'users.npy' for en in embedding_files]
        )
    else:
        user_embeddings = None

    target = conf['target']
    train_target = train_data[target].values
    val_target = val_data[target].values

    train_ds = ufm.DS(user_feature_loader, user_embeddings, target, train_data['user_id'], train_target)
    val_ds = ufm.DS(user_feature_loader, user_embeddings, target, val_data['user_id'], val_target)
    model_class = ufm.models[conf['model_class']]

    model = model_class(
        conf_to_dict(conf['model_hyperparameters']), target, user_feature_loader.n_cat_features, run_dir)
    model.fit(train_ds, val_ds)
    del train_ds
    model.save(run_dir)

    val_ds_no_target = ufm.DS(user_feature_loader, user_embeddings, target, val_data['user_id'])
    val_prediction = model.predict_probability(val_ds_no_target)
    np.save(str(run_dir / 'val_prediction.npy'), val_prediction)
    del user_embeddings

    test_data = pd.read_parquet(feature_utils.DATA_ROOT / 'submit_2.pqt')
    test_users = test_data['user_id'].values
    user_embeddings = feature_utils.PartialEmbeddingsLoader(
        test_users,
        [feature_utils.EMBEDDINGS_DIR / en / 'users.npy' for en in embedding_files]
    )
    test_ds = ufm.DS(user_feature_loader, user_embeddings, target, test_users)
    test_prediction = model.predict_probability(test_ds)
    np.save(str(run_dir / 'test_prediction.npy'), test_prediction)


if __name__ == '__main__':
    main()
