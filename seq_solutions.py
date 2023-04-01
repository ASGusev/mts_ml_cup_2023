from itertools import chain

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, f1_score


v_len = np.vectorize(len)


def batch_seqs(seqs, len_limit):
    longest_seq = max(seqs, key=len)
    max_len = min(len(longest_seq), len_limit)
    # noinspection PyUnresolvedReferences
    batch = np.full((len(seqs), *longest_seq.shape), -1, seqs[0].dtype)
    for i, s in enumerate(seqs):
        batch[i, :min(max_len, len(s))] = s[:max_len]
    return batch


class UsersDataset(Dataset):
    def __init__(
            self,
            users, targets,
            users_histories, interaction_features, users_cat_features, users_num_features, user_embeddings,
            urls_embeddings, urls_features,
            batch_size, hist_len_limit, order_by_len=True
    ):
        valid_targets_mask = ~np.isnan(targets.ravel())
        histories = users_histories[valid_targets_mask]
        self.batch_size = batch_size
        self.hist_len_limit = hist_len_limit
        histories_lengths = v_len(histories)
        users_order = np.argsort(histories_lengths) if order_by_len else np.arange(len(histories_lengths))
        self.users = users[valid_targets_mask][users_order]
        self.histories = histories[users_order]
        self.interaction_features = interaction_features[valid_targets_mask][users_order]
        self.targets = targets[valid_targets_mask][users_order]
        self.users_cat_features = users_cat_features[valid_targets_mask][users_order].copy()
        self.users_num_features = users_num_features[valid_targets_mask][users_order].copy()
        if user_embeddings is not None:
            self.user_embeddings = user_embeddings[valid_targets_mask][users_order].copy()
        else:
            self.user_embeddings = None
        self.urls_embeddings = [
            np.concatenate([emb, np.zeros((1, emb.shape[1]))])
            for emb in urls_embeddings
        ]
        self.urls_features = np.concatenate([urls_features, np.zeros((1, urls_features.shape[1]))])

    def __len__(self):
        return (len(self.users) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, item):
        batch_start_index = item * self.batch_size
        batch_end_index = (item + 1) * self.batch_size
        batch_slice = slice(batch_start_index, batch_end_index)
        batch_users_cat_features = self.users_cat_features[batch_slice]
        batch_users_num_features = self.users_num_features[batch_slice]
        if self.user_embeddings is not None:
            batch_users_num_features = np.concatenate(
                [batch_users_num_features, self.user_embeddings[batch_slice]],
                axis=1
            )
        batch_targets = self.targets[batch_slice]
        batch_raw_histories = self.histories[batch_slice]
        batch_histories = batch_seqs(batch_raw_histories, self.hist_len_limit)
        batch_histories_embeddings = [emb[batch_histories] for emb in self.urls_embeddings]
        interaction_features = batch_seqs(self.interaction_features[batch_slice], self.hist_len_limit)
        batch_histories_masks = batch_histories >= 0
        batch_histories_urls_features = self.urls_features[batch_histories]
        return {
            'user_cat_features': torch.tensor(batch_users_cat_features),
            'user_num_features': torch.tensor(batch_users_num_features),
            'history_embeddings': list(map(torch.tensor, batch_histories_embeddings)),
            'interaction_features': torch.tensor(interaction_features),
            'history_url_features': torch.tensor(batch_histories_urls_features),
            'history_mask': torch.tensor(batch_histories_masks),
            'target': torch.tensor(batch_targets)
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# noinspection PyAbstractClass
class DatasetShuffleWrapper(IterableDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        order = np.random.permutation(len(self))
        for i in order:
            yield self.dataset[i]


class Gini:
    name = 'gini'

    def __call__(self, targets, predictions):
        return 2 * roc_auc_score(targets, predictions) - 1


class WeightedF1:
    name = 'weighted_f1'

    def __call__(self, targets, predictions):
        prediction_tops = np.argmax(predictions, axis=1)
        return f1_score(targets, prediction_tops, average='weighted')


class Attention(nn.Module):
    def __init__(self, input_dim, temperature=1., n_layers=1):
        super().__init__()
        self.temperature = temperature
        self.att_conv = nn.Sequential(
            *chain.from_iterable(
                (
                    nn.BatchNorm1d(input_dim),
                    nn.Conv1d(input_dim, input_dim, 1),
                    nn.PReLU(),
                )
                for _ in range(n_layers - 1)
            ),
            nn.BatchNorm1d(input_dim),
            nn.Conv1d(input_dim, 1, 1)
        )

    def forward(self, features, mask):
        att_vals = torch.exp(self.att_conv(features) / self.temperature)
        att_vals = att_vals * mask
        att_weights = att_vals / att_vals.sum(dim=2, keepdim=True)
        return torch.sum(features * att_weights, dim=2)


class ConvModel(pl.LightningModule):
    def __init__(
            self,
            url_embeddings_dims=(256,), user_cat_features_cardinalities=(80, 984, 37, 599, 4), cat_embeddings_dim=1,
            user_num_features_dim=24, user_embeddings_dims=(), url_features_dim=1, interaction_features_dim=1,
            history_hidden_dim=256, conv_kernel=1, n_conv_layers=1, combined_hidden_dim=256, n_final_steps=2,
            attention=False, attention_temperature=1., n_attention_layers=1, dropout=.2, out_dim=1,
            lr=1e-2, weight_decay=1e-3, class_weights=None
    ):
        super().__init__()
        self.save_hyperparameters()

        def make_embeddings_path(embeddings_dim):
            history_input_dim = embeddings_dim + url_features_dim + interaction_features_dim + 1
            conv_layers = [
                nn.BatchNorm1d(history_input_dim),
                nn.Dropout(dropout),
                nn.Conv1d(history_input_dim, history_hidden_dim, conv_kernel, padding=conv_kernel // 2),
                nn.PReLU(),
            ]
            for _ in range(1, n_conv_layers):
                conv_layers.append(nn.BatchNorm1d(history_hidden_dim))
                conv_layers.append(nn.Dropout(dropout))
                conv_layers.append(
                    nn.Conv1d(history_hidden_dim, history_hidden_dim, conv_kernel, padding=conv_kernel // 2))
                conv_layers.append(nn.PReLU())
            return nn.Sequential(*conv_layers)

        self.history_paths = nn.ModuleList(map(make_embeddings_path, url_embeddings_dims))
        self.cat_features_embeddings = nn.ModuleList(
            nn.Embedding(fc, cat_embeddings_dim) for fc in user_cat_features_cardinalities
        )
        if attention:
            self.attentions = nn.ModuleList(
                Attention(history_hidden_dim, attention_temperature, n_attention_layers)
                for _ in url_embeddings_dims
            )
        else:
            self.attentions = None
        final_clf_input_dim = len(url_embeddings_dims) * history_hidden_dim + user_num_features_dim + \
            sum(user_embeddings_dims) + len(user_cat_features_cardinalities) * cat_embeddings_dim
        self.clf_part = nn.Sequential(
            nn.BatchNorm1d(final_clf_input_dim),
            nn.Dropout(dropout),
            nn.Linear(final_clf_input_dim, combined_hidden_dim),
            nn.PReLU(),

            *chain.from_iterable(
                (
                    nn.BatchNorm1d(combined_hidden_dim),
                    nn.Dropout(dropout),
                    nn.Linear(combined_hidden_dim, combined_hidden_dim),
                    nn.PReLU()
                )
                for _ in range(n_final_steps - 2)
            ),

            nn.BatchNorm1d(combined_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(combined_hidden_dim, out_dim),
            nn.Sigmoid() if out_dim == 1 else nn.Identity()
        )
        if class_weights:
            class_weights = torch.tensor(class_weights)
        self.loss_function = nn.BCELoss() if out_dim == 1 else nn.CrossEntropyLoss(weight=class_weights)
        self.val_metric = Gini() if out_dim == 1 else WeightedF1()
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, batch):
        history_mask = batch['history_mask'].unsqueeze(1)
        history_inputs = [
            torch.concatenate(
                (
                    emb.permute((0, 2, 1)),
                    batch['history_url_features'].permute((0, 2, 1)),
                    batch['interaction_features'].permute((0, 2, 1)),
                    history_mask
                ),
                dim=1
            ).float()
            for emb in batch['history_embeddings']
        ]
        history_hidden = [hp(hi) for hp, hi in zip(self.history_paths, history_inputs)]
        if self.attentions:
            history_int_res = [attention(hh, history_mask) for attention, hh in zip(self.attentions, history_hidden)]
        else:
            mask_sum = history_mask.sum(dim=2)
            history_int_res = [(hh * history_mask).sum(dim=2) / mask_sum for hh in history_hidden]
        cat_features_embeddings = [
            emb(batch['user_cat_features'][:, i])
            for i, emb in enumerate(self.cat_features_embeddings)
        ]
        combined_features = torch.concatenate(
            (
                *history_int_res,
                *cat_features_embeddings,
                batch['user_num_features']
            ),
            dim=1
        ).float()
        return self.clf_part(combined_features)

    def configure_optimizers(self):
        return torch.optim.NAdam(self.parameters(), self.lr, weight_decay=self.weight_decay)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch):
        prediction = self(batch)
        loss = self.loss_function(prediction, batch['target'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, _):
        with torch.no_grad():
            prediction = self(batch)
        loss = self.loss_function(prediction, batch['target'])
        self.log('validation_loss', loss)
        return prediction.cpu(), batch['target'].cpu()

    def validation_epoch_end(self, outputs):
        predictions, targets = [], []
        for p, t in outputs:
            predictions.append(p.numpy())
            targets.append(t.numpy())
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        self.log(self.val_metric.name, self.val_metric(targets, predictions))
