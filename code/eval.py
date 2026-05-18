"""
# Experiments

# Experiment 0 - Hyperparameter Tuning on Experiment 1


# Experiment 1 – Learning parameter to quality characteristics mapping

Here we need to compare the distance metric of the models for inferring quality characteristics given an unseen testset.
This includes:
    + Evaluating different models. (MODEL TYPE)
    + Training on eight seeds and evaluating on eight seeds. (MODEL STABILITY)
    + Evaluating on different test sets (mode = 'start', 'middle', 'end', 'random'). (MODEL EXPRESSIVITY - CAN IT EXTRAPOLATE)
Here it is totally ok to just report the distances with percentiles for the different combinations as table, as only the result counts mainly




# Experiment 2 – Evaluating optimization Algorithms for Gradient-based Process Step Parameter Search

Here we evaluate different optimizers that are used in paramopt, from simple stochastic gradient descent to RMSprop.
This includes:
    + Evaluating different models
    + Evaluating different optimizers
    + Parameter Reconstruction with the same model on eight different seeds (due to the stochasticity of the optimization process).
    + Evaluation of the whole test set and then averaged (as table?)
    + Evaluation of different constellations and numbers of reconstructed variables.
Here I want to show some optimization curves for the different optimizers (averaged and with some confidence bound around them).




# Experiment 3 – Benchmarking Gradient-based Process Step Parameter Estimation with Uninformed and Heuristic Search

Here we evaluate the performance of paramopt against the two other search paradimgs, uninformed search and semi informed search.
This includes:
    + Evaluating different methods (paramopt_wb, paramopt_bb, beamsearch, genetic_algorithm).
    + Evaluating different models.
    + Evaluating on all samples from the test set.
    + Evalauting on different seeds due to the stochasticity of the optimization process.
    + Evaluating on different starting regimes for the parameter guesses (zeros, ones, (random for multistart optimization)).
    + Evaluating on different numbers and constellations of reconstructed variables.
Here I want to have a table with final results and confidence bounds, as well as figures that compare the convergence process.


# Experiment 4 – Evaluating Model Transferability

Here we evaluate the transferability of pretrained models on different test datasets.
This includes:
    + Evaluating each model on the test sets of all other datasets than the one it was trained on
    + Including eight different seeds due to the stochasticity of the optimization process
    + Including different starting regimes for the parameter guesses (zeros, ones, (random for multistart optimization)).
    + Evaluating different numbers and constellations of reconstructed variables.
Here I want to have a table with confidence bounds.



# Experiment design:

FOR EACH MODEL   ('ff', 'res', 'mdn', 'hres', 'tabpfn', 'autotabpfn') (6)
    FOR EACH SEED    (1-8) (8)
        FOR EACH DATASET    (1-8) (8)
            FOR EACH SELECTION_MODE_TEST_SET   (start; middle; end; random) (4)
                TRAIN A MODEL ---------------------------------------------------------------------------------------------- LOG_EXP1: LIST(TRAIN_ERRORS), LIST(VAL_ERRORS), LIST(TEST_ERRORS)

                FOR ESTIMATION METHOD (paramopt; beamsearch; genetic_algorithm) (3)
                    IF paramopt:
                        FOR EACH OTHER DATASET TESTSET (4 tors, 4 cont) (4)
                            FOR EACH OPTIMIZER (sgd, sgd-nesterov, adam, asgd, rmsprop) (5)
                                FOR EACH CONSTELLATION OF RECONSTRUCTION SCENARIO (tff; ftf; ttt; ...) (7)
                                    FOR EACH STARTING REGIME (zeros; guess; random) (2)
                                        OPTIMIZE INPUT PARAMETERS OF THE TEST SET ------------------------------------------ LOG_EXP2: LIST OF PARAM VALUES DURING RECONSTRUCTION AND START_PARAM AND GOAL_PARAM
                                                                                                                             LOG_EXP3: LIST OF PARAM VALUES DURING RECONSTRUCTION AND START_PARAM AND GOAL_PARAM
                                                                                                                             LOG_EXP4: LIST OF PARAM VALUES DURING RECONSTRUCTION AND START_PARAM AND GOAL_PARAM
                    ELSE:
                        FOR EACH OTHER DATASET TESTSET (4 tors, 4 cont) (4)
                            FOR EACH CONSTELLATION OF RECONSTRUCTION SCENARIO (tff; ftf; ttt; ...) (7)
                                FOR EACH STARTING REGIME (zeros; ones; random) (3)
                                    OPTIMIZE INPUT PARAMETERS OF THE TEST SET -------------------------------------------------- LOG_EXP3: LIST OF PARAM VALUES DURING RECONSTRUCTION AND START_PARAM AND GOAL_PARAM
"""

import json

from train import *
from opt import *

import argparse


class ExperimentRun():
    def __init__(self, model_type:str, seed:int, dataset_id:str, testset_selection:str, parameter_estimation_method:str):
        # init hyperparameters
        self.hparam = self._init_hparam(model_type, seed, dataset_id, testset_selection)

        # init directories for experiments
        (self.save_path_exp1,
         self.save_path_exp2,
         self.save_path_exp3,
         self.save_path_exp4) = self._init_dirs(model_type, dataset_id, testset_selection)

        # train model - experiment 1
        self.model = self.train_model(hparam=self.hparam, save_path=self.save_path_exp1)

        # do the other experiments with the trained model
        hparam = self.hparam.copy()
        self.run_experiments(hparam=hparam, model=model_type, parameter_estimation_method=parameter_estimation_method)

    def _init_hparam(self, model_type:str, seed:int, dataset_id:str, testset_selection:str):
        # quick workaround because of batchsize
        if dataset_id == 'ds5':
            batch_size = 1
        else:
            batch_size = 4

        hparam = {
            "ID": f"{model_type}_{dataset_id}_{testset_selection}_{seed}",
            "DS_ID": f"{dataset_id}",
            "SEED": seed,
            "DATA_DIR": f"../data/{dataset_id}/usw.csv",
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

            "METHOD": "is",

            "FOR_OPT_PARAMS": [True, False, False],
            "OPT": "SGD",
            "OPT_LR": 0.05,
            "OPT_MOMENTUM": 0.1,
            "OPT_THRESHOLD": 0.1,
            "OPT_MAX_CYCLES": 500,
            "OPT_MAX_CON_CYCLES": 200,
            "OPT_PATIENCE": 25}
        return hparam

    def _init_dirs(self, model_type:str, dataset_id:str, testset_selection:str):
        # dirs for exp1
        save_path_exp1 = f'../exp_paper/exp1/{dataset_id}/{model_type}/{testset_selection}'
        os.makedirs(save_path_exp1, exist_ok=True)

        # dirs for exp2
        save_path_exp2 = f'../exp_paper/exp2/{dataset_id}/{model_type}/{testset_selection}'
        os.makedirs(save_path_exp2, exist_ok=True)

        # dirs for exp3
        save_path_exp3 = f'../exp_paper/exp3/{dataset_id}/{model_type}/{testset_selection}'
        os.makedirs(save_path_exp3, exist_ok=True)

        # dirs for exp4
        save_path_exp4 = f'../exp_paper/exp4/{dataset_id}/{model_type}/{testset_selection}'
        os.makedirs(save_path_exp4, exist_ok=True)

        return save_path_exp1, save_path_exp2, save_path_exp3, save_path_exp4

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

        :param model_type: which model shall be trained: ff, res, mdn, hres, tabpfn, autotabpfn
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
        if hparam['DS_ID'] != dataset_id:
            hparam['DS_ID'] = dataset_id
            hparam['DATA_DIR'] = f"../data/{dataset_id}/usw.csv"

        data_module = DataModuleUsw(hparam=hparam, modus=hparam['TESTSET_SELECTION'], scaling=False)
        dl_test = data_module.get_test_dataloader()

        logging_x_ground_truth = []
        logging_y_ground_truth = []
        logging_x_guess = []
        logging_y_prediction = []

        i = 0
        for x_gt, y_gt in dl_test:
            if i > 7:
                break
            else:
                opt_module = OptModule(hparam=hparam, model=self.model)
                x_guess = self._init_parameter_guesses(x_gt=x_gt, mask=parameter_constellation, method=parameter_initialization)

                (logging_x_ground_truth_sample,
                 logging_y_ground_truth_sample,
                 logging_x_guess_sample,
                 logging_y_prediction_sample) = opt_module.find_params(x_guess=x_guess, y_guess=y_gt, x_gt=x_gt, y_gt=y_gt, opt_vars=parameter_constellation)

                logging_x_ground_truth.append(logging_x_ground_truth_sample)
                logging_y_ground_truth.append(logging_y_ground_truth_sample)
                logging_x_guess.append(logging_x_guess_sample)
                logging_y_prediction.append(logging_y_prediction_sample)
            i += 1

        results_dict = {
            "ID": "",
            "HPARAMS": hparam,
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

        optimizers = ['SGD', 'SGD-nesterov', 'Adam', 'ASGD', 'RMSprop']
        parameter_constellations = [[False, True, True], [False, False, True], [False, True, False], [True, False, False],
                                    [True, True, False], [True, False, True], [True, True, True]]
        parameter_initializations = ['random']
        ds_group1 = ['ds1', 'ds3', 'ds5', 'ds7']
        ds_group2 = ['ds2', 'ds4', 'ds6', 'ds8']

        if parameter_estimation_method == 'is':
            for optimizer in optimizers:
                hparam['OPT'] = optimizer

                for parameter_constellation in parameter_constellations:
                    hparam['FOR_OPT_PARAMS'] = parameter_constellation

                    for parameter_initialization in parameter_initializations:
                        hparam['OPT_INIT'] = parameter_initialization

                        if hparam['DS_ID'] in ds_group1:
                            for ds in ds_group1:
                                results_dict = self.experiment(hparam=hparam, dataset_id=ds, parameter_constellation=parameter_constellation, parameter_initialization=parameter_initialization)

                                if hparam['DS_ID'] == ds:
                                    # logging for exp 2 and exp 3
                                    results_dict['ID'] = 'exp2'
                                    self._save_results(hparam=hparam, results_dict=results_dict, save_path=self.save_path_exp2, dataset_id=f'{ds}_{parameter_constellation}_{parameter_estimation_method}_{optimizer}')
                                    results_dict['ID'] = 'exp3'
                                    self._save_results(hparam=hparam, results_dict=results_dict, save_path=self.save_path_exp3, dataset_id=f'{ds}_{parameter_constellation}_{parameter_estimation_method}_{optimizer}')
                                else:
                                    # logging for exp 4
                                    results_dict['ID'] = 'exp4'
                                    self._save_results(hparam=hparam, results_dict=results_dict, save_path=self.save_path_exp4, dataset_id=f'{ds}_{parameter_constellation}_{parameter_estimation_method}_{optimizer}')
                        else:
                            for ds in ds_group2:
                                results_dict = self.experiment(hparam=hparam, dataset_id=ds, parameter_constellation=parameter_constellation, parameter_initialization=parameter_initialization)

                                if hparam['DS_ID'] == ds:
                                    # logging for exp 2 and exp 3
                                    results_dict['ID'] = 'exp2'
                                    self._save_results(hparam=hparam, results_dict=results_dict, save_path=self.save_path_exp2, dataset_id=f'{ds}_{parameter_constellation}_{parameter_estimation_method}_{optimizer}')
                                    results_dict['ID'] = 'exp3'
                                    self._save_results(hparam=hparam, results_dict=results_dict, save_path=self.save_path_exp3, dataset_id=f'{ds}_{parameter_constellation}_{parameter_estimation_method}_{optimizer}')
                                else:
                                    # logging for exp 4
                                    results_dict['ID'] = 'exp4'
                                    self._save_results(hparam=hparam, results_dict=results_dict,  save_path=self.save_path_exp4, dataset_id=f'{ds}_{parameter_constellation}_{parameter_estimation_method}_{optimizer}')

        else:
            for parameter_constellation in parameter_constellations:
                hparam['FOR_OPT_PARAMS'] = parameter_constellation

                for parameter_initialization in parameter_initializations:
                    hparam['OPT_INIT'] = parameter_initialization

                    results_dict = self.experiment(hparam=hparam, dataset_id=hparam['DS_ID'], parameter_constellation=parameter_constellation, parameter_initialization=parameter_initialization)

                    # logging for exp 3
                    results_dict['ID'] = 'exp3'
                    self._save_results(hparam=hparam, results_dict=results_dict, save_path=self.save_path_exp3, dataset_id=f'{parameter_constellation}_{parameter_estimation_method}')


        if isinstance(self.model, AutoTabPFNRegressor) and hasattr(self.model, "predictor_"):
            try:
                predictor_path = self.model.predictor_.path
                try:
                    self.model.predictor_.save_space()
                except Exception:
                    pass
                shutil.rmtree(predictor_path, ignore_errors=True)
            except Exception as e:
                print(f"Warning: could not clean AutoTabPFN models: {e}")

        return




if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--estimation_method", required=True)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--testset_selection", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ExperimentRun(
        model_type=args.model_type,
        seed=args.seed,
        dataset_id=args.dataset_id,
        testset_selection=args.testset_selection,
        parameter_estimation_method=args.estimation_method,
    )
    

    model_types = ['ff', 'res', 'hres', 'autotabpfn']
    estimation_methods = ['is', 'sis', 'us']
    dataset_ids = ['ds1', 'ds2', 'ds3', 'ds4', 'ds5', 'ds7', 'ds8']
    testset_selections = ['start', 'end', 'random']
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    seed = 0

    for model_type in model_types:
        for estimation_method in estimation_methods:
            for dataset_id in dataset_ids:
                for testset_selection in testset_selections:
                    exp_run_class = ExperimentRun(model_type, seed, dataset_id, testset_selection, estimation_method)

    """
    """
    # model_tpye x esimation_method x dataset_id x test_set_selection x seed


    model_type = 'ff'
    seed = 0
    dataset_id = 'ds1'
    testset_selection = 'start'
    parameter_estimation_method = 'is'


    exp_run_class = ExperimentRun(model_type, seed, dataset_id, testset_selection, parameter_estimation_method)
    """

    model_types = ['autotabpfn'] # ['hres', 'autotabpfn'] # ['ff', 'res'] #
    estimation_methods = ['is', 'sis', 'us']
    dataset_ids = ['ds1', 'ds2', 'ds3', 'ds4', 'ds5', 'ds7', 'ds8']
    testset_selections = ['end'] #, ['start', 'end', 'random']
    seeds = [0] #, 1, 2, 3, 4, 5, 6, 7, 8]

    for model_type in model_types:
        for seed in seeds:
            for dataset_id in dataset_ids:
                for testset_selection in testset_selections:
                    for estimation_method in estimation_methods:
                        exp_run_class = ExperimentRun(model_type, seed, dataset_id, testset_selection, estimation_method)