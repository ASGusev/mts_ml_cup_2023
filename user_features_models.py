import abc
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from omegaconf import DictConfig, OmegaConf
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score, f1_score

from feature_utils import FeatureLoader, PartialEmbeddingsLoader, age_bucket


class DS:
    def __init__(
            self,
            user_feature_loader: FeatureLoader, user_embeddings: PartialEmbeddingsLoader, target_type: str,
            user_ids: np.ndarray, target_values: Optional[np.ndarray] = None
    ):
        if target_values is not None:
            if target_type == 'is_male':
                target_values = target_values.astype(np.float32)
            mask = ~np.isnan(target_values)
            self.user_ids = user_ids[mask]
            self.target = target_values[mask]
            if target_type == 'age':
                self.target = age_bucket(self.target)
        else:
            self.user_ids = user_ids
            self.target = None
        self.user_feature_loader = user_feature_loader
        self.user_embeddings = user_embeddings

    def get_user_cat_features(self):
        return self.user_feature_loader.get_cat_features(self.user_ids)

    def get_user_num_features(self):
        num_features = self.user_feature_loader.get_num_features(self.user_ids)
        if not self.user_embeddings:
            return num_features
        embeddings = self.user_embeddings[self.user_ids]
        return np.concatenate((num_features, embeddings), axis=1)

    def get_all_user_features(self):
        cat_features = self.user_feature_loader.get_cat_features(self.user_ids)
        num_features = self.user_feature_loader.get_num_features(self.user_ids)
        embeddings = self.user_embeddings[self.user_ids]
        return np.concatenate((cat_features, num_features, embeddings), axis=1)


class AbstractModel(abc.ABC):
    def __init__(self, model_hps, target, n_cat_features, train_dir=None):
        self.model_hps = model_hps
        self.target = target
        self.cat_feature_ids = np.arange(n_cat_features)
        self.train_dir = train_dir

    @abc.abstractmethod
    def fit(self, train_ds: DS, val_ds: DS):
        pass

    @abc.abstractmethod
    def predict_probability(self, ds: DS):
        pass

    @abc.abstractmethod
    def save(self, directory: Path):
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, path: Path, conf: DictConfig):
        pass


class GBDTModel(AbstractModel):

    def __init__(self, model_hps, target, n_cat_features, train_dir=None, model=None):
        super().__init__(model_hps, target, n_cat_features, train_dir)
        self.cat_feature_names = [f'c{i}' for i in self.cat_feature_ids]
        self.model = model or CatBoostClassifier(
            **model_hps['create'], cat_features=self.cat_feature_names,
            eval_metric='AUC' if target == 'is_male' else 'TotalF1',
            train_dir=train_dir / 'catboost_info' if train_dir else None
        )

    def prepare_features(self, ds: DS):
        cat_features = ds.get_user_cat_features()
        num_features = ds.get_user_num_features()
        return pd.DataFrame({
            **{fn: fvs for fn, fvs in zip(self.cat_feature_names, cat_features.T)},
            **{f'n{i:04}': fvs for i, fvs in enumerate(num_features.T)}
        })

    def fit(self, train_ds: DS, val_ds: DS):
        self.model.fit(
            self.prepare_features(train_ds), train_ds.target,
            eval_set=(self.prepare_features(val_ds), val_ds.target) if val_ds else None
        )

    def predict_probability(self, ds: DS):
        probability = self.model.predict_proba(self.prepare_features(ds))
        if self.target == 'is_male':
            probability = probability[:, 1]
        return probability

    def save(self, directory: Path):
        self.model.save_model(str(directory / 'model.cbm'))

    @classmethod
    def load(cls, path: Path, conf: DictConfig):
        model = CatBoostClassifier()
        model.load_model(str(path / 'model.cbm'))
        return GBDTModel(conf['model_parameters'], conf['target'], len(conf['features']['cat_features']), model=model)


class Gini(Metric):
    def __init__(self):
        self._name = "gini"
        self._maximize = True

    def __call__(self, y_true, y_score):
        auc = roc_auc_score(y_true, y_score[:, 1])
        return max(2 * auc - 1, 0.)


class WeightedF1(Metric):
    def __init__(self):
        self._name = "weighted_f1"
        self._maximize = True

    def __call__(self, y_true, y_score):
        return f1_score(y_true, np.argmax(y_score, axis=1), average='weighted')


class NNModel(AbstractModel):
    def __init__(self, model_hps, target, n_cat_features, train_dir=None, model=None):
        super().__init__(model_hps, target, n_cat_features, train_dir)
        self.model = model

    def fit(self, train_ds: DS, val_ds: DS):
        cat_cardinalities = train_ds.get_user_cat_features().max(axis=0) + 1
        # noinspection PyTypeChecker
        self.model = TabNetClassifier(
            cat_idxs=self.cat_feature_ids.tolist(), cat_emb_dim=np.ones_like(self.cat_feature_ids).tolist(),
            cat_dims=cat_cardinalities.tolist(), **self.model_hps['create']
        )
        # noinspection PyTypeChecker
        self.model.fit(
            train_ds.get_all_user_features(), np.vectorize(str)(train_ds.target),
            eval_set=[(val_ds.get_all_user_features(), np.vectorize(str)(val_ds.target))],
            eval_metric=[Gini if self.target == 'is_male' else WeightedF1],
            **self.model_hps['fit']
        )

    def predict_probability(self, ds: DS):
        probability = self.model.predict_proba(ds.get_all_user_features())
        if self.target == 'is_male':
            probability = probability[:, 1]
        return probability

    def save(self, directory: Path):
        self.model.save_model(str(directory / 'model'))

    @classmethod
    def load(cls, path: Path, conf: DictConfig):
        model = TabNetClassifier()
        model.load_model(str(path / 'model.zip'))
        return NNModel(conf['model_parameters'], conf['target'], len(conf['features']['cat_features']), model=model)


models = {
    'gbdt': GBDTModel,
    'nn': NNModel
}


def load_from_run_dir(run_dir: Path):
    conf = OmegaConf.load(run_dir / '.hydra' / 'config.yaml')
    model_class = models[conf['model_class']]
    return model_class.load(run_dir, conf)
