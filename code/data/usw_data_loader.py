from .generic_data_loader import *
from sklearn.model_selection import KFold


class DataModuleUsw(DataModule):
    def __init__(self, hparam, modus, scaling=False):
        super(DataModuleUsw, self).__init__(hparam=hparam)
        data = self.load_data()
        ds = self.clean(data)
        if hparam['N_AUG_SAMPLES'] > 0:
            ds = self.augment_ds(ds, n_add_samples=hparam['N_AUG_SAMPLES'], categorical_channels=[], noise_fraction=0.1)

        if scaling:
            ds = self.scaler(ds=ds)
        else:
            ds = ds.to_numpy()

        self.ds_train, self.ds_test = self.sampler(self.to_tensor(ds), modus)
        #self.kfold = KFold(n_splits=hparam['K_FOLDS'], shuffle=True, random_state=42)

    def clean(self, ds):
        columns = ['Amplitude', 'Schweißzeit', 'Vorschub', 'Schweißdruck', 'Schweißkraft', 'Zugscherkraft']
        available_cols = [col for col in columns if col in ds.columns]
        ds = ds[available_cols].copy()
        if 'Schweißzeit' in available_cols:
            ds.loc[ds['Schweißzeit'] == 'zeitgeregelt', 'Schweißzeit'] = ds['Schweißzeit'].mode()[0]
        ds = ds.apply(pd.to_numeric, errors='coerce')

        ds = ds[(ds >= 0).all(axis=1)]
        ds = ds.dropna()
        return ds

    def xy_split(self, ds):
        x = ds[:, :-1]
        y = ds[:, -1].unsqueeze(dim=1)
        dataset = TensorDataset(x, y)
        return dataset

    def get_fold_dataloaders(self, fold_index):
        fold_indices = list(self.kfold.split(self.ds_train))
        train_idx, val_idx = fold_indices[fold_index-1]
        ds_train = self.ds_train[train_idx]
        ds_val = self.ds_train[val_idx]

        train_loader = self.init_dataloader(self.xy_split(ds_train), batch_size=self.hparam['BATCH_SIZE'], shuffle=True)
        val_loader = self.init_dataloader(self.xy_split(ds_val), batch_size=self.hparam['BATCH_SIZE'], shuffle=False)
        return train_loader, val_loader

    def get_train_dataloaders(self, train_val_ratio=0.9):
        n_total = len(self.ds_train)
        n_train = int(n_total * train_val_ratio)    #round(len(self.ds_train) * train_val_ratio)
        n_val   = n_total - n_train     #round(len(self.ds_train) * (1 - train_val_ratio))

        ds_train, ds_val = random_split(self.ds_train, [n_train, n_val])
        train_loader = self.init_dataloader(ds_train, batch_size=self.hparam['BATCH_SIZE'], shuffle=True)
        val_loader   = self.init_dataloader(ds_val,  batch_size=self.hparam['BATCH_SIZE'], shuffle=False)
        return train_loader, val_loader

    def get_test_dataloader(self):
        test_loader = self.init_dataloader(self.ds_test, batch_size=1, shuffle=False)
        return test_loader


# quick test and quick run
if __name__ == "__main__":
    hparam = {
        "ID": "ds1_hparam",
        "DS_ID": "ds8",
        "SEED": 42,
        "DATA_DIR": "../../data/ds8/usw.csv",
        "STUDY_DIR": "../../exp/exp1",
        "LOG_DIR": "../../exp/exp1/ds8",
        "K_FOLDS": 4,
        "N_AUG_SAMPLES": 0,
        "BATCH_SIZE": 16,

        "INPUT_DIM": 3,
        "HIDDEN_DIM": 16,
        "OUTPUT_DIM": 1,
        "N_LAYERS": 4,
        "DROPOUT": 0.1,

        "MAX_EPOCHS": 100,
        "LR": 0.005,
        "WEIGHT_DECAY": 0.0001,

        "METHOD": "is",
        "FOR_OPT_PARAMS": [False, False, True],
        "OPT": "SGD",
        "OPT_LR": 0.05,
        "OPT_MOMENTUM": 0.1,
        "OPT_THRESHOLD": 0.1
    }

    data_module = DataModuleUsw(hparam=hparam, modus='end')

    train_loader, val_loader = data_module.get_train_dataloaders()
    test_loader = data_module.get_test_dataloader()

    train_x, train_y = next(iter(train_loader))
    val_x, val_y = next(iter(val_loader))
    test_x, test_y = next(iter(test_loader))

    print("format: torch.Tensor(batch, channels, seq_len: ")
    print("train_x shape: ", train_x.shape)
    print("train_y shape: ", train_y.shape)
    print("val_x shape: ", val_x.shape)
    print("val_y shape: ", val_y.shape)
    print("test_x shape: ", test_x.shape)
    print("test_y shape: ", test_y.shape)
    print(50*"-")