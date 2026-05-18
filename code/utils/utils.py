import os
import torch
import joblib
import numpy as np

def create_dir(path, dir_name):
    path = os.path.join(path, dir_name)

    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} was created.')

def launch_tensorboard(log_dir):
    os.system('tensorboard --logdir="' + log_dir + '"')
    return

def to_tensor(array):
    return torch.tensor(array, dtype=torch.float32)


def xy_transform(hparam, x, y):
    scaler = joblib.load(hparam['LOG_DIR'] + '/scaler.gz')

    arr = np.append(x, y).reshape(1, -1)
    arr = scaler.transform(arr)
    arr = arr[0]

    x, y = np.hsplit(arr, [hparam['SPLIT_LOC']])
    return x, y


def xy_inverse_transform(hparam, x, y):
    scaler = joblib.load(hparam['LOG_DIR'] + '/scaler.gz')

    arr = np.append(x, y).reshape(1, -1)
    arr = scaler.inverse_transform(arr)
    arr = arr[0]

    x, y = np.hsplit(arr, [hparam['SPLIT_LOC']])
    return x, y


def init_hparam(dataset='ds1', model_type='res', seed=1, experiment_id=1, data_batch_size=1,
                model_hidden_dim=128,
                model_n_layers=8, model_dropout=0.2, model_max_epochs=500, model_lr=5e-4, model_wd=1e-4,
                opt_method='is',
                opt_params=[True, False, False], opt_optimizer='SGD', opt_lr=0.05, opt_momentum=0.1,
                opt_threshold=0.1,
                opt_max_cycles=1000, opt_max_con_cycles=200, opt_patience=25):
    hparam = {}

    hparam['ID'] = f'{dataset}_{model_type}_{seed}'
    hparam['DS_ID'] = dataset
    hparam['SEED'] = seed
    hparam['DEVICE'] = 0
    hparam['DATA_DIR'] = f'../data/{dataset}/usw.csv'
    hparam['STUDY_DIR'] = f'../exp/exp{experiment_id}{dataset}/{model_type}/'
    hparam['LOG_DIR'] = f'../exp/exp{experiment_id}/{dataset}/{model_type}/{seed}/'
    hparam['MODEL_TYPE'] = model_type
    hparam['N_AUG_SAMPLES'] = 0
    hparam['BATCH_SIZE'] = data_batch_size
    hparam['INPUT_DIM'] = 3
    hparam['OUTPUT_DIM'] = 1
    hparam['HIDDEN_DIM'] = model_hidden_dim
    hparam['N_LAYERS'] = model_n_layers
    hparam['DROPOUT'] = model_dropout
    hparam['MAX_EPOCHS'] = model_max_epochs
    hparam['LR'] = model_lr
    hparam['WEIGHT_DECAY'] = model_wd

    hparam['METHOD'] = opt_method

    hparam['FOR_OPT_PARAMS'] = opt_params
    hparam['OPT'] = opt_optimizer
    hparam['OPT_LR'] = opt_lr
    hparam['OPT_MOMENTUM'] = opt_momentum
    hparam['OPT_THRESHOLD'] = opt_threshold
    hparam['OPT_MAX_CYCLES'] = opt_max_cycles
    hparam['OPT_MAX_CON_CYCLES'] = opt_max_con_cycles
    hparam['OPT_PATIENCE'] = opt_patience

    return hparam
