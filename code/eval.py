"""Reproducible computational evaluations for the paramOpt paper.

The baseline suite reproduces Table 2's comparison on the six datasets.  The
gradient suite reproduces Table 3's MLP/Res/HRes/TabPFN comparison on dsRS1.
Each uses eight independent seeds, all selected test samples, a 1,000-step
budget, and the four parameter masks reported in the manuscript.
"""

import json
import os
from pathlib import Path

import torch

from train import *
from opt import *

import argparse


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_PARAMETER_CONSTELLATIONS = [
    [True, False, False],       # amplitude
    [False, True, False],       # time / feed rate
    [False, True, True],        # time / feed rate and force
    [True, True, True],         # all machine parameters
]


class ExperimentRun():
    def __init__(self, model_type:str, seed:int, dataset_id:str, testset_selection:str,
                 parameter_estimation_method:str, optimization_objective:str = 'target_match'):
        # init hyperparameters
        self.hparam = self._init_hparam(
            model_type, seed, dataset_id, testset_selection, optimization_objective
        )

        # init directories for experiments
        self.save_path_exp1, self.save_path_exp3 = self._init_dirs(
            model_type, dataset_id, testset_selection
        )

        # train model - experiment 1
        self.model = self.train_model(hparam=self.hparam, save_path=self.save_path_exp1)

        # do the other experiments with the trained model
        hparam = self.hparam.copy()
        self.run_experiments(hparam=hparam, model=model_type, parameter_estimation_method=parameter_estimation_method)

    def _init_hparam(self, model_type:str, seed:int, dataset_id:str, testset_selection:str,
                     optimization_objective:str):
        # quick workaround because of batchsize
        if dataset_id == 'ds5':
            batch_size = 1
        else:
            batch_size = 4

        hparam = {
            "ID": f"{model_type}_{dataset_id}_{testset_selection}_{seed}",
            "DS_ID": f"{dataset_id}",
            "SEED": seed,
            "DATA_DIR": str(REPO_ROOT / 'data' / dataset_id / 'usw.csv'),
            "MODEL_TYPE": f"{model_type}",
            "TESTSET_SELECTION": f"{testset_selection}",
            "BATCH_SIZE": batch_size,
            "INPUT_DIM": 3,
            "OUTPUT_DIM": 1,
            "HIDDEN_DIM": 128,
            "N_LAYERS": 8,
            "DROPOUT": 0.1,
            "MAX_EPOCHS": 1000,  # TODO change back to #400,
            "LR": 0.005,
            "WEIGHT_DECAY": 0.0001,
            "N_AUG_SAMPLES": 0,

            "METHOD": "paramopt",
            "OPT_OBJECTIVE": optimization_objective,

            "FOR_OPT_PARAMS": [True, False, False],
            "OPT": "RMSprop",
            "OPT_LR": 0.05,
            "OPT_MOMENTUM": 0.1,
            "OPT_THRESHOLD": 0.1,
            "OPT_MAX_CYCLES": 1000,
            "OPT_MAX_CON_CYCLES": 200,
            "OPT_EARLY_STOP": False,
            "OPT_RESTARTS": 1,
            "OPT_PATIENCE": 25}
        return hparam

    def _init_dirs(self, model_type:str, dataset_id:str, testset_selection:str):
        # dirs for exp1
        save_path_exp1 = REPO_ROOT / 'results' / 'paper' / 'exp1' / dataset_id / model_type / testset_selection
        save_path_exp1.mkdir(parents=True, exist_ok=True)

        # dirs for exp3
        save_path_exp3 = REPO_ROOT / 'results' / 'paper' / 'exp3' / dataset_id / model_type / testset_selection
        save_path_exp3.mkdir(parents=True, exist_ok=True)

        return str(save_path_exp1), str(save_path_exp3)

    def _save_results(self, hparam:dict, results_dict:dict, save_path:str, dataset_id=None):
        file_name = hparam['ID']

        if dataset_id is not None:
            file_path = os.path.join(save_path, f'{file_name}_on_{dataset_id}.json')
        else:
            file_path = os.path.join(save_path, f'{file_name}.json')

        with open(file_path, 'w') as f:
            json.dump(results_dict, f)
        return print(f'Results saved to {file_path}')

    def _init_parameter_guesses(self, x_gt, mask, method='zeros'):
        if not isinstance(x_gt, torch.Tensor):
            x_gt = torch.tensor(x_gt, dtype=torch.float, device=x_gt.device)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool, device=x_gt.device)

        if method == 'zeros':
            x = torch.zeros_like(x_gt)
        elif method == 'random':
            x = torch.rand_like(x_gt) * x_gt

        x_guess = torch.where(mask, x, x_gt)
        return x_guess

    def _tensor_to_list(self, obj):
        """Recursively convert torch.Tensors in obj to (nested) Python lists."""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()  # or just obj.tolist() if already on CPU
        elif isinstance(obj, list):
            return [self._tensor_to_list(x) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(self._tensor_to_list(x) for x in obj)
        elif isinstance(obj, dict):
            return {k: self._tensor_to_list(v) for k, v in obj.items()}
        else:
            return obj

    def train_model(self, hparam:dict, save_path:str):
        """
        Effectively experiment 1 ... comparing how well the models fit on the available data.

        :param model_type: which model shall be trained: ff, res, mdn, hres, or tabpfn
        :param seed: fixed seed for reproducibility
        :param dataset: which dataset shall be used: ds1, ds2, ds3 ...
        :param testset_selection: from where in the tabular data shall the testset be selected from: start, middle, end, random ?
        """

        training_class = TrainModule(hparam=hparam, modus=hparam['TESTSET_SELECTION'], scaling=False)
        results_dict = training_class.training()

        # logging for exp 1
        self._save_results(hparam=hparam, results_dict=results_dict, save_path=save_path)

        model = training_class.model
        return model

    def experiment(self, hparam:dict, dataset_id:str, parameter_constellation:list, parameter_initialization:str):
        hparam = hparam.copy()
        hparam['DS_ID'] = dataset_id
        hparam['DATA_DIR'] = str(REPO_ROOT / 'data' / dataset_id / 'usw.csv')

        data_module = DataModuleUsw(hparam=hparam, modus=hparam['TESTSET_SELECTION'], scaling=False)
        dl_test = data_module.get_test_dataloader()

        logging_x_ground_truth = []
        logging_y_ground_truth = []
        logging_x_guess = []
        logging_y_prediction = []

        for x_gt, y_gt in dl_test:
            opt_module = OptModule(hparam=hparam, model=self.model)
            x_guess = self._init_parameter_guesses(x_gt=x_gt, mask=parameter_constellation, method=parameter_initialization)

            y_target = y_gt if hparam['OPT_OBJECTIVE'] == 'target_match' else None
            (logging_x_ground_truth_sample,
             logging_y_ground_truth_sample,
             logging_x_guess_sample,
             logging_y_prediction_sample) = opt_module.find_params(
                x_guess=x_guess,
                x_gt=x_gt,
                y_gt=y_target,
                opt_vars=parameter_constellation,
            )

            logging_x_ground_truth.append(logging_x_ground_truth_sample)
            logging_y_ground_truth.append(logging_y_ground_truth_sample)
            logging_x_guess.append(logging_x_guess_sample)
            logging_y_prediction.append(logging_y_prediction_sample)

        results_dict = {
            "ID": "",
            "HPARAMS": hparam,
            "OPT_OBJECTIVE": hparam['OPT_OBJECTIVE'],
            "X_GROUND_TRUTH": logging_x_ground_truth,
            "Y_GROUND_TRUTH": logging_y_ground_truth,
            "X_GUESS": logging_x_guess,
            "Y_PREDICTIONS": logging_y_prediction,
            "LEN_TESTSET": len(dl_test),
        }

        results_dict = self._tensor_to_list(results_dict)

        return results_dict

    def run_experiments(self, hparam:dict, model:str, parameter_estimation_method:str):
        hparam['MODEL_TYPE'] = model
        hparam['METHOD'] = parameter_estimation_method

        parameter_constellations = PAPER_PARAMETER_CONSTELLATIONS
        parameter_initializations = ['random']
        source_dataset_id = hparam['DS_ID']
        for parameter_constellation in parameter_constellations:
            hparam['FOR_OPT_PARAMS'] = parameter_constellation

            for parameter_initialization in parameter_initializations:
                hparam['OPT_INIT'] = parameter_initialization
                results_dict = self.experiment(
                    hparam=hparam,
                    dataset_id=source_dataset_id,
                    parameter_constellation=parameter_constellation,
                    parameter_initialization=parameter_initialization,
                )
                results_dict['ID'] = 'exp3'

                if parameter_estimation_method == 'paramopt':
                    result_id = (
                        f'{source_dataset_id}_{parameter_constellation}_'
                        f'{parameter_estimation_method}_{hparam["OPT"]}'
                    )
                else:
                    result_id = f'{parameter_constellation}_{parameter_estimation_method}'

                self._save_results(
                    hparam=hparam,
                    results_dict=results_dict,
                    save_path=self.save_path_exp3,
                    dataset_id=result_id,
                )


        return




def run_paper_suite(suite: str, testset_selection: str = 'end'):
    """Execute the two computational evaluations reported in the paper.

    The physical open-loop validation is intentionally not included here: it
    requires a welding experiment and must not be represented as a software
    result.
    """
    seeds = range(8)
    if suite == 'baselines':
        configurations = [
            ('res', dataset_id, method)
            for dataset_id in ['ds1', 'ds2', 'ds3', 'ds4', 'ds5', 'ds6']
            for method in ['paramopt', 'beam_search', 'genetic_algorithm']
        ]
    elif suite == 'gradients':
        configurations = [
            (model_type, 'ds1', 'paramopt')
            for model_type in ['ff', 'res', 'hres', 'tabpfn']
        ]
    else:
        raise ValueError(f'Unknown paper suite: {suite}')

    for model_type, dataset_id, method in configurations:
        for seed in seeds:
            ExperimentRun(
                model_type=model_type,
                seed=seed,
                dataset_id=dataset_id,
                testset_selection=testset_selection,
                parameter_estimation_method=method,
                optimization_objective='target_match',
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run paramOpt evaluations.')
    parser.add_argument('--suite', choices=['baselines', 'gradients'])
    parser.add_argument('--model-type', choices=['ff', 'res', 'hres', 'tabpfn'])
    parser.add_argument('--estimation-method', choices=['paramopt', 'beam_search', 'genetic_algorithm'])
    parser.add_argument('--dataset-id', choices=['ds1', 'ds2', 'ds3', 'ds4', 'ds5', 'ds6'])
    parser.add_argument('--testset-selection', choices=['start', 'middle', 'end', 'random'], default='end')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--objective', choices=sorted(VALID_OBJECTIVES), default='target_match')
    args = parser.parse_args()

    if args.suite:
        run_paper_suite(args.suite, args.testset_selection)
    elif args.model_type and args.estimation_method and args.dataset_id:
        ExperimentRun(
            model_type=args.model_type,
            seed=args.seed,
            dataset_id=args.dataset_id,
            testset_selection=args.testset_selection,
            parameter_estimation_method=args.estimation_method,
            optimization_objective=args.objective,
        )
    else:
        parser.error('Specify --suite or all of --model-type, --estimation-method, and --dataset-id.')
