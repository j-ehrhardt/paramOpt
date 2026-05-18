import os
import ast
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pad_run(run, L):
    a = np.asarray(run)
    if a.ndim == 2 and a.shape[-1] == 3:
        a = a[:, None, :]

    if a.shape[0] == 0:
        return np.zeros((L, 1, 3), dtype=float)

    k = L - a.shape[0]
    if k > 0:
        a = np.concatenate([a, np.repeat(a[-1:], k, axis=0)], axis=0)
    return a


def pad_samples(sample_list, L):
    return np.stack([np.stack([pad_run(run, L) for run in sample], axis=0) for sample in sample_list], axis=0)


def retrieve_optimization_results_same_ds(model_type:str, dataset_id:str, testset_selection:str, optimization_setup:str, paradigm):
    filepath = f'../../exp_hpc/exp3/{dataset_id}/{model_type}/{testset_selection}'

    if paradigm == 'is':
        filename = f'{model_type}_{dataset_id}_{testset_selection}_0_on_{dataset_id}_{optimization_setup}_is_RMSprop.json'
    else:
        filename = f'{model_type}_{dataset_id}_{testset_selection}_0_on_{optimization_setup}_{paradigm}.json'


    file = os.path.join(filepath, filename)

    with open(file, 'r') as f:
        results = json.load(f)

    x_gt_list = results['X_GROUND_TRUTH']
    x_guess_list = results['X_GUESS']

    #max_L = max(max(len(s[0]) for s in x_gt_list), max(len(s[0]) for s in x_guess_list))
    #max(max(len(s[0]) for s in x_gt_list), max(len(s[0]) for s in x_guess_list)))

    #x_gt = np.stack([np.pad(np.asarray(s),((0, 0), (0, 0), (0, max_L - len(s[0])), (0, 0), (0, 0)), mode="edge") for s in x_gt_list])
    #x_guess = np.stack([np.pad(np.asarray(s),((0, 0), (0, 0), (0, max_L - len(s[0])), (0, 0), (0, 0)), mode="edge") for s in x_guess_list])

    max_L = max(max(len(s[0]) for s in x_gt_list), max(len(s[0]) for s in x_guess_list))

    x_gt = pad_samples(x_gt_list, max_L)
    x_guess = pad_samples(x_guess_list, max_L)

    """ very old
    #x_gt = np.array(x_gt)
    #x_guess = np.array(x_guess)

    #x_gt = np.array(x_gt).squeeze(axis=3)
    #x_guess = np.array(x_guess).squeeze(axis=3)
    """

    x_gt = x_gt.squeeze(axis=3)
    x_guess = x_guess.squeeze(axis=3)

    return x_gt, x_guess



def build_mae_table_per_setup_with_sd(model_type: str, dataset_ids: list[str], testset_selection: str, setups: list[str], paradigms: list[str], ddof: int = 1, decimals: int = 1, format_cells: bool = True, verbose: bool = False):
    """
    Rows: (setup, optimizer)
    Cols: ds1..ds8

    Each cell summarizes final-step MAE over sample×run:
      - mean ± sd  (if format_cells=True)
      - OR numeric columns ds_mean / ds_sd (if format_cells=False)

    MAE is computed as mean(|x_guess - x_gt|) over selected parameter dims (mask), then final step.
    """
    def setup_mask(setup_str: str) -> np.ndarray:
        return np.array(ast.literal_eval(setup_str), dtype=bool)

    def final_step_mae_samples(x_gt: np.ndarray, x_guess: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # x_gt, x_guess: (S,R,T,D)
        err = np.abs(x_guess - x_gt)  # (S,R,T,D)
        D = err.shape[-1]
        mask = mask[:D]
        if mask.sum() == 0:
            return np.array([np.nan])

        err = err[..., mask]              # (S,R,T,D_sel)
        mae_per_step = err.mean(axis=-1)  # (S,R,T)
        final_mae = mae_per_step[..., -1] # (S,R)
        return final_mae.reshape(-1)      # (S*R,)

    rows = []

    for setup in setups:
        mask = setup_mask(setup)

        for par in paradigms:
            row = {"setup": setup, "paradigm": par}

            for ds in dataset_ids:
                try:
                    x_gt, x_guess = retrieve_optimization_results_same_ds(
                        model_type=model_type,
                        dataset_id=ds,
                        testset_selection=testset_selection,
                        optimization_setup=setup,
                        paradigm=par,
                    )
                    samples = final_step_mae_samples(x_gt, x_guess, mask)
                    m = float(np.nanmean(samples))
                    s = float(np.nanstd(samples, ddof=ddof))

                    if format_cells:
                        row[ds] = "--" if np.isnan(m) else f"{m:.{decimals}f} ± {s:.{decimals}f}"
                    else:
                        row[f"{ds}_mean"] = m
                        row[f"{ds}_sd"] = s

                except Exception as e:
                    if format_cells:
                        row[ds] = "--"
                    else:
                        row[f"{ds}_mean"] = np.nan
                        row[f"{ds}_sd"] = np.nan
                    if verbose:
                        print(f"[missing] model={model_type}, ds={ds}, sel={testset_selection}, setup={setup}, par={par} -> {e}")

            rows.append(row)

    df = pd.DataFrame(rows).set_index(["setup", "paradigm"])
    return df



if __name__ == '__main__':

    dataset_ids = ['ds1', 'ds3', 'ds5', 'ds4', 'ds6', 'ds8']
    setups = ['[True, False, False]', '[False, True, False]', '[False, True, True]', '[True, True, True]']
    paradigms = ['is'] #, 'sis', 'us']

    df1 = build_mae_table_per_setup_with_sd(model_type='ff', dataset_ids=dataset_ids, testset_selection='end', setups=setups, paradigms=paradigms)
    df2 = build_mae_table_per_setup_with_sd(model_type='res', dataset_ids=dataset_ids, testset_selection='end', setups=setups, paradigms=paradigms)
    df3 = build_mae_table_per_setup_with_sd(model_type='hres', dataset_ids=dataset_ids, testset_selection='end', setups=setups, paradigms=paradigms)
    df4 = build_mae_table_per_setup_with_sd(model_type='autotabpfn', dataset_ids=dataset_ids, testset_selection='end', setups=setups, paradigms=paradigms)


    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200, "display.expand_frame_repr", False):
            print('========ff========')
            print(df1)
            print('========res========')
            print(df2)
            print('========hres========')
            print(df3)
            print('========autotabpfn========')
            print(df4)

