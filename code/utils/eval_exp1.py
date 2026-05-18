"""
# Experiment 1 – Learning parameter to quality characteristics mapping

Here we need to compare the distance metric of the models for inferring quality characteristics given an unseen testset.
This includes:
    + Evaluating different models. (MODEL TYPE)
    + Training on eight seeds and evaluating on eight seeds. (MODEL STABILITY)
    + Evaluating on different test sets (mode = 'start', 'middle', 'end', 'random'). (MODEL EXPRESSIVITY - CAN IT EXTRAPOLATE)
Here it is totally ok to just report the distances with percentiles for the different combinations as table, as only the result counts mainly
"""

import os
import json
import numpy as np
import pandas as pd


def compute_training_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    err = y_pred - y_true
    se = err ** 2
    ae = np.abs(err)

    mse = se.mean()
    rmse = float(np.sqrt(mse))
    mae = ae.mean()

    ss_res = se.sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return float(mse), float(rmse), float(mae), float(r2)


def retrieve_training_results(model_type: str, dataset_id: str, testset_selection: str, seeds: list):
    filepath = f'../../exp_hpc/exp1/{dataset_id}/{model_type}/{testset_selection}/'

    mses, rmses, maes, r2s = [], [], [], []

    for seed in seeds:
        filename = f'{model_type}_{dataset_id}_{testset_selection}_{seed}.json'
        file = os.path.join(filepath, filename)

        try:
            with open(file, 'r') as f:
                results_dict = json.load(f)

            if model_type == 'hres':
                y_true = results_dict['TEST_GROUND_TRUTHS']
                y_pred = [elem[0] for elem in  results_dict['TEST_PREDICTIONS']]
            else:
                y_true = results_dict['TEST_GROUND_TRUTHS']
                y_pred = results_dict['TEST_PREDICTIONS']

            mse, rmse, mae, r2 = compute_training_metrics(y_true, y_pred)

            mses.append(mse)
            rmses.append(rmse)
            maes.append(mae)
            r2s.append(r2)

        except FileNotFoundError:
            print(f'No file found for {filename}')
            continue
        except Exception as e:
            print(f"Error for seed {seed} ({file}): {e}")
            continue

    if len(mses) == 0:
        raise RuntimeError("No valid runs found for the given seeds.")

    mses = np.asarray(mses, dtype=float)
    rmses = np.asarray(rmses, dtype=float)
    maes = np.asarray(maes, dtype=float)
    r2s  = np.asarray(r2s,  dtype=float)

    metrics = {
        "mse": {
            "mean": float(mses.mean()),
            "std":  float(mses.std(ddof=1)) if mses.size > 1 else 0.0,
            "per_seed": mses,
        },
        "rmse": {
            "mean": float(rmses.mean()),
            "std":  float(rmses.std(ddof=1)) if rmses.size > 1 else 0.0,
            "per_seed": rmses,
        },
        "mae": {
            "mean": float(maes.mean()),
            "std":  float(maes.std(ddof=1)) if maes.size > 1 else 0.0,
            "per_seed": maes,
        },
        "r2": {
            "mean": float(r2s.mean()),
            "std":  float(r2s.std(ddof=1)) if r2s.size > 1 else 0.0,
            "per_seed": r2s,
        },
    }

    return metrics


def convert_to_training_table(model_types:list, dataset_ids:list, testset_selections:list, seeds:list, metrics:list):
    df_cumulative = pd.DataFrame()

    for metric in metrics:
        rows = []

        # collect per-seed metric values
        for model_type in model_types:
            for dataset_id in dataset_ids:
                for testset_selection in testset_selections:
                    try:
                        results = retrieve_training_results(model_type, dataset_id, testset_selection, seeds)
                    except RuntimeError:
                        # no valid runs for this combo
                        continue

                    for metric_value in results[metric]["per_seed"]:
                        rows.append({
                            "model": model_type,
                            "dataset": dataset_id,
                            "testset_selection": testset_selection,
                            metric: float(metric_value),
                        })

        df = pd.DataFrame(rows)

        for sel in testset_selections:
            df_sel = df[df["testset_selection"] == sel]
            if df_sel.empty:
                continue

            agg_sel = df_sel.groupby(["model", "dataset"])[metric].agg(["mean", "std"])
            agg_sel["mean_sd"] = agg_sel.apply(
                lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}",
                axis=1
            )

            tmp = agg_sel.reset_index()
            tmp["metric"] = metric
            tmp["testset_selection"] = sel
            df_cumulative = pd.concat([df_cumulative, tmp], ignore_index=True)


            table_sel = agg_sel["mean_sd"].unstack("dataset")

            print(f"\n===== {metric} (testset_selection = {sel}) =====")
            print(table_sel.to_string())
    return df_cumulative


if __name__ == '__main__':
    model_types = ['ff', 'res', 'hres'] #, 'autotabpfn']
    dataset_ids = ['ds1', 'ds2', 'ds3', 'ds4', 'ds5', 'ds6', 'ds7', 'ds8']
    testset_selections = ['end'] #['start', 'end', 'random']
    metrics = ['mae', 'mse', 'rmse', 'r2']
    seeds = [0, 1, 2, 3, 4, 5, 6, 7]

    df = convert_to_training_table(model_types, dataset_ids, testset_selections, seeds, metrics)