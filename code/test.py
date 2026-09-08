import os
import torch

from train import *
from opt import *
import json


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def init_guesses(x_gt, mask, method='zeros'):
    if not isinstance(x_gt, torch.Tensor):
        x_gt = torch.tensor(x_gt, dtype=torch.float, device=x_gt.device)
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.bool, device=x_gt.device)

    if method == 'zeros':
        x = torch.zeros_like(x_gt)
    elif method == 'random':
        x = torch.rand_like(x_gt) * x_gt
    elif method == 'values':
        x = torch.full_like(x_gt, fill_value=1.0)

    x_guess = torch.where(mask, x, x_gt)
    return x_guess

def exp_run(hparam):
    train_class = TrainModule(hparam=hparam, modus='random', scaling=False)

    training_results = train_class.training()
    model = train_class.model

    data_module = DataModuleUsw(hparam=hparam, modus='random', scaling=False)
    test_data = data_module.get_test_dataloader()

    results = {}
    results['data'] = hparam['DATA_DIR']

    opt_scenarios = [[True, False, False], [False, True, False], [False, False, True], [True, True, False], [True, False, True], [False, True, True], [True, True, True]]

    for scenario in opt_scenarios:
        results[str(scenario)] = {}
        i = 0
        for x_gt_batch, y_gt_batch in test_data:
            for x_gt, y_gt in zip(x_gt_batch, y_gt_batch):

                opt_class = OptModule(hparam=hparam, model=model)
                x_guess = init_guesses(x_gt=x_gt, mask=scenario, method='zeros')

                x_gt_history, y_gt_history, x_hat_history, y_hat_history = opt_class.find_params(
                    x_guess=x_guess,
                    x_gt=x_gt,
                    y_gt=y_gt,
                    opt_vars=scenario,
                )
                results[str(scenario)][i] = {
                    'x_gt': x_gt.tolist(),
                    'x_hat': x_hat_history,
                    'y_gt': y_gt.tolist(),
                    'y_hat': y_hat_history,
                    'training_test_loss': training_results['TEST_LOSSES_AVG'],
                }
                i+= 1

        print(results)

    save_path = f'../results/{hparam["MODEL_TYPE"]}/{hparam["DS_ID"]}'
    os.makedirs(save_path, exist_ok=True)
    with open(f'{save_path}/results.json', 'w') as f:
        json.dump(to_jsonable(results), f, indent=4)


if __name__ == '__main__':
    hparam = {
        "ID": "ds",
        "DS_ID": "ds1",
        "SEED": 42,
        "DEVICE": 0,
        "DATA_DIR": "../data/ds1/usw.csv",
        "STUDY_DIR": "../exp/exp1",
        "LOG_DIR": "../exp/exp1/ds1",
        'MODEL_TYPE': 'tabpfn',  # res, hres, mdn, ff, tabpfn
        "N_AUG_SAMPLES": 0,
        "BATCH_SIZE": 4,
        "INPUT_DIM": 3,
        "OUTPUT_DIM": 1,
        "HIDDEN_DIM": 128,
        "N_LAYERS": 8,
        "DROPOUT": 0.2,
        "MAX_EPOCHS": 1,
        "LR": 0.0005,
        "WEIGHT_DECAY": 0.0001,

        "METHOD": "paramopt",
        "OPT_OBJECTIVE": "target_match",

        "FOR_OPT_PARAMS": [True, False, False],
        "OPT": "SGD",
        "OPT_LR": 0.05,
        "OPT_MOMENTUM": 0.1,
        "OPT_THRESHOLD": 0.1,
        "OPT_MAX_CYCLES": 200,
        "OPT_MAX_CON_CYCLES": 200,
        "OPT_PATIENCE": 25,
    }

    #exp_run(hparam=hparam)


    for model in ['ff', 'res', 'mdn', 'hres', 'tabpfn']:
        hparam['MODEL_TYPE'] = model

        hparam = hparam.copy()
        exp_run(hparam=hparam)
    """"""
