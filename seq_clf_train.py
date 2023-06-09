import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import hydra
import omegaconf

import feature_utils
import seq_solutions


@hydra.main(version_base=None, config_path='.', config_name='run_conf.yaml')
def main(conf: omegaconf.DictConfig):
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

    def make_ds(data):
        user_ids = data['user_id'].values
        target = data[conf['target']].values
        if conf['target'] == 'is_male':
            target = target.astype(np.float32).reshape((-1, 1))
        else:
            target = feature_utils.age_bucket(target).astype(np.int64)
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
    if conf['shuffle_train_ds']:
        train_ds = seq_solutions.DatasetShuffleWrapper(train_ds)
    val_ds = make_ds(val_data)

    class_weights = None
    if conf['target'] == 'is_male':
        out_dim = 1
        val_metric = 'gini'
    else:
        out_dim = feature_utils.N_AGE_BUCKETS
        val_metric = 'weighted_f1'
        if conf['weight_classes']:
            class_weights = np.zeros(feature_utils.N_AGE_BUCKETS)
            classes = feature_utils.age_bucket(train_data['age']).astype(np.int32)
            np.add.at(class_weights, classes, 1)
            class_weights = class_weights / class_weights.sum()
            class_weights = class_weights.tolist()
    model = seq_solutions.ConvModel(**conf['model_parameters'], out_dim=out_dim, class_weights=class_weights)
    pb_callback = pl.callbacks.TQDMProgressBar(refresh_rate=16)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, monitor=val_metric, mode='max')
    trainer = pl.Trainer(**conf['trainer_parameters'], callbacks=[pb_callback, checkpoint_callback])
    # noinspection PyTypeChecker
    trainer.fit(model, train_ds, val_ds)


if __name__ == '__main__':
    main()
