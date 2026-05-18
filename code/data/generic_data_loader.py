import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import joblib


class DataModule(nn.Module):
    def __init__(self, hparam):
        super(DataModule, self).__init__()
        self.hparam = hparam

    def load_data(self):
        data = pd.read_csv(self.hparam['DATA_DIR'], delimiter='\t')
        return data

    def augment_ds(self, ds, n_add_samples=50, categorical_channels=[], noise_fraction=0.5): #0.1):
        # Make a copy to avoid modifying original ds
        ds = ds.copy()
        # Create an order column based on current order
        ds['order'] = np.arange(len(ds))

        categorical_channels = [col for col in categorical_channels if col in ds.columns]
        categorical_df = ds[categorical_channels].copy()
        numerical_df = ds.drop(columns=categorical_channels + ['order'])

        augmented_data = []

        for _ in range(n_add_samples):
            # Sample a single row (keeping its index)
            sample = numerical_df.sample(n=1)
            sample_index = sample.index[0]
            # Create noise and add it to the numerical part
            noise = noise_fraction * pd.Series(np.random.randn(*sample.shape).reshape(-1))
            noisy_sample = sample + noise.values

            # If there are categorical channels, get them from the original sample
            if not categorical_df.empty:
                original_cat = categorical_df.loc[[sample_index]].reset_index(drop=True)
                noisy_sample = noisy_sample.reset_index(drop=True)
                augmented_sample = pd.concat([original_cat, noisy_sample], axis=1)
            else:
                augmented_sample = noisy_sample.reset_index(drop=True)

            # Assign an order value slightly higher than the original sample’s order
            orig_order = ds.loc[sample_index, 'order']
            augmented_sample['order'] = orig_order + 0.1  # ensures augmented sample comes just after original
            augmented_data.append(augmented_sample)

        augmented_df = pd.concat(augmented_data, ignore_index=True)
        combined = pd.concat([ds, augmented_df], ignore_index=True)
        combined = combined.sort_values('order').reset_index(drop=True)
        combined = combined.drop(columns='order')

        return combined

    def to_tensor(self, ds):
        ds = torch.tensor(ds, dtype=torch.float32)
        return ds

    def scaler(self, ds):
        Scaler = StandardScaler()
        ds = Scaler.fit_transform(ds)

        if not os.path.exists(self.hparam['LOG_DIR']):
            os.makedirs(self.hparam['LOG_DIR'])

        joblib.dump(Scaler, filename=self.hparam['LOG_DIR'] + '/scaler.gz')
        return ds

    def sampler(self, data, modus):
        # The sampler function cuts out data from the front, the middle, or the back of the DoE
        # this allows for testing the extrapolation capabilities of the pre-trained models.

        # split paramters and effect into x and y values
        x, y = data[:, :-1], data[:, -1]

        n_samples = data.shape[0]
        train_size = int(0.9 * n_samples)
        test_size = n_samples - train_size

        indices = torch.arange(start=0, end=n_samples, step=1, dtype=torch.int32)

        if modus == 'start':
            train_indices = indices[test_size:]
            test_indices = indices[:test_size]
        elif modus == 'middle':
            train_indices = torch.cat((indices[:train_size//2], indices[(train_size//2)+test_size:]))
            test_indices = indices[train_size//2:(-train_size//2)]
        elif modus == 'end':
            train_indices = indices[:-test_size]
            test_indices = indices[-test_size:]
        elif modus == 'random':
            permuted_indices = indices[torch.randperm(len(indices))]
            test_indices = permuted_indices[:test_size]
            train_indices = permuted_indices[test_size:]

        ds_train = TensorDataset(x[train_indices], y[train_indices])
        ds_test = TensorDataset(x[test_indices], y[test_indices])
        return ds_train, ds_test

    def init_dataloader(self, ds, batch_size=1, shuffle=False):
        return DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=shuffle, drop_last=True, pin_memory=True)




