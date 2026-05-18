"""
# Experiment 2 – Evaluating optimization Algorithms for Gradient-based Process Step Parameter Search

Here we evaluate different optimizers that are used in paramopt, from simple stochastic gradient descent to RMSprop.
This includes:
    + Evaluating different models
    + Evaluating different optimizers
    + Parameter Reconstruction with the same model on eight different seeds (due to the stochasticity of the optimization process).
    + Evaluation of the whole test set and then averaged (as table?)
    + Evaluation of different constellations and numbers of reconstructed variables.
Here I want to show some optimization curves for the different optimizers (averaged and with some confidence bound around them).
"""

import os
import json
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle

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


def retrieve_optimization_results_same_ds(model_type:str, dataset_id:str, testset_selection:str, optimization_setup:str, optimizer:str):
    #filepath = f'../../exp_paper/exp2_hpc/{dataset_id}/{model_type}/{testset_selection}'
    #filename = f'{model_type}_{dataset_id}_{testset_selection}_0_on_{dataset_id}_{optimization_setup}_{optimizer}.json'

    filepath = f'../../exp_hpc/exp2/{dataset_id}/{model_type}/{testset_selection}'
    filename = f'{model_type}_{dataset_id}_{testset_selection}_0_on_{dataset_id}_{optimization_setup}_is_{optimizer}.json'

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


def aggregate_optimization_metrics(model_type:str, dataset_id:str, testset_selection:str, optimization_setup:str, optimizers:list):

    results = {}

    for optimizer in optimizers:
        try:
            x_gt, x_guess = retrieve_optimization_results_same_ds(model_type, dataset_id, testset_selection, optimization_setup, optimizer)

            results[optimizer] = {}
            results[optimizer]['X_GROUND_TRUTH'] = np.array(x_gt)
            results[optimizer]['X_GUESS'] = np.array(x_guess)

            x_loss = np.absolute(x_guess - x_gt).sum(axis=-1)
            results[optimizer]['X_LOSS'] = x_loss

            x_loss_mean = x_loss.mean(axis=-2)
            x_loss_var = x_loss.var(axis=-2)
            x_loss_sd = x_loss.std(axis=-2)

            results[optimizer]['X_LOSS_MEAN'] = x_loss_mean
        except:
            continue
    return results


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def _valid_savgol_params(n, window_length, polyorder):
    # make window odd
    window_length = int(window_length)
    if window_length % 2 == 0:
        window_length += 1
    # cap to length and keep odd
    window_length = min(window_length, n if n % 2 == 1 else n - 1)
    window_length = max(window_length, 3)  # minimum usable
    polyorder = int(polyorder)
    polyorder = min(polyorder, window_length - 1)
    return window_length, polyorder


from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import ast

def plot_optimization_results_same_ds_per_optimizer(
    model_type,
    dataset_id,
    testset_selection,
    optimization_setup,
    optimizers,
    max_steps: int | None = None,     # NEW: clip number of steps (e.g., 200)
    headline: str | None = None,      # NEW: custom title
    smooth: bool = True,
    sg_window: int = 21,
    sg_poly: int = 3,
    show_band: bool = False,
    band_alpha: float = 0.12,
    legend_alpha: float = 0.85,       # legend box transparency
):
    mask = np.array(ast.literal_eval(optimization_setup), dtype=bool)

    fig, ax = plt.subplots(figsize=(6, 4))

    for opt in optimizers:
        try:
            x_gt, x_guess = retrieve_optimization_results_same_ds(
                model_type, dataset_id, testset_selection, optimization_setup, opt
            )
        except Exception:
            continue

        # x_gt/x_guess: (S,R,T,D)
        err = np.abs(x_guess - x_gt)
        D = err.shape[-1]
        m = mask[:D]
        err = err[..., m]                        # (S,R,T,D_sel)

        mae = err.mean(axis=-1)                  # (S,R,T)
        mae_sr = mae.reshape(-1, mae.shape[-1])  # (S*R, T)

        mean = mae_sr.mean(axis=0)
        sd   = mae_sr.std(axis=0, ddof=1) if mae_sr.shape[0] > 1 else np.zeros_like(mean)

        # NEW: clip steps
        if max_steps is not None:
            Tclip = min(int(max_steps), len(mean))
            mean = mean[:Tclip]
            sd   = sd[:Tclip]

        # smoothing (after clipping so window checks are correct)
        if smooth and len(mean) >= 3:
            wl = int(sg_window)
            if wl % 2 == 0:
                wl += 1
            wl = min(wl, len(mean) if len(mean) % 2 == 1 else len(mean) - 1)
            wl = max(wl, 3)
            po = min(int(sg_poly), wl - 1)

            mean = savgol_filter(mean, window_length=wl, polyorder=po)
            sd   = np.maximum(savgol_filter(sd, window_length=wl, polyorder=po), 0.0)

        steps = np.arange(len(mean))
        line, = ax.plot(steps, mean, linewidth=2.6, label=opt)

        if show_band and mae_sr.shape[0] > 1:
            ax.fill_between(
                steps, mean - sd, mean + sd,
                alpha=band_alpha, color=line.get_color(), linewidth=0
            )

    # NEW: custom headline
    if headline is None:
        headline = f"{dataset_id} · {model_type} · {optimization_setup}"
    ax.set_title(headline)

    ax.set_xlabel("optimization steps")
    ax.set_ylabel("MAE")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # NEW: legend style (top-right, semi-transparent box)
    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=legend_alpha,
        borderpad=0.6,
    )

    fig.tight_layout()
    #fig.savefig(f"{headline}.pdf", bbox_inches="tight")
    plt.show()

"""
def plot_optimization_results_same_ds_per_optimizer_(
    model_type, dataset_id, testset_selection, optimization_setup, optimizers,
    smooth=True, sg_window=21, sg_poly=3,
    show_band=False, band_alpha=0.05
):
    mask = np.array(ast.literal_eval(optimization_setup), dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for opt in optimizers:
        try:
            x_gt, x_guess = retrieve_optimization_results_same_ds(
                model_type, dataset_id, testset_selection, optimization_setup, opt
            )
        except Exception:
            continue

        # x_gt/x_guess: (S,R,T,D)
        err = np.abs(x_guess - x_gt)
        D = err.shape[-1]
        m = mask[:D]
        err = err[..., m]                       # (S,R,T,D_sel)

        mae = err.mean(axis=-1)                 # (S,R,T)
        mae_sr = mae.reshape(-1, mae.shape[-1]) # (S*R, T)

        mean = mae_sr.mean(axis=0)
        sd   = mae_sr.std(axis=0, ddof=1) if mae_sr.shape[0] > 1 else np.zeros_like(mean)

        if smooth and len(mean) >= 3:
            wl = sg_window + (sg_window % 2 == 0)  # ensure odd
            wl = min(wl, len(mean) if len(mean) % 2 == 1 else len(mean) - 1)
            wl = max(wl, 3)
            po = min(sg_poly, wl - 1)
            mean = savgol_filter(mean, wl, po)
            sd   = np.maximum(savgol_filter(sd, wl, po), 0.0)

        steps = np.arange(len(mean))
        line, = ax.plot(steps, mean, linewidth=2.6, label=opt)

        if show_band and mae_sr.shape[0] > 1:
            ax.fill_between(steps, mean - sd, mean + sd, alpha=band_alpha,
                            color=line.get_color(), linewidth=0)

    ax.set_title(f"{dataset_id} · {model_type} · {optimization_setup}")
    ax.set_xlabel("optimization steps")
    ax.set_ylabel("MAE (masked dims)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def plot_optimization_results_same_ds_per_optimizer_(
    model_type: str,
    dataset_id: str,
    testset_selection: str,
    optimization_setup: str,
    optimizers: list,
    smooth: bool = True,
    sg_window: int = 21,
    sg_poly: int = 3,
    show_band: bool = False,
    band_alpha: float = 0.12,
):
    results = aggregate_optimization_metrics(
        model_type, dataset_id, testset_selection, optimization_setup, optimizers
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for optimizer, stats in results.items():
        x = np.asarray(stats["X_LOSS_MEAN"])  # (runs, steps) typically
        mean = x.mean(axis=0)

        # std (FIX vs your code: sd is sqrt(var), not var)
        sd = x.std(axis=0, ddof=1) if x.shape[0] > 1 else np.zeros_like(mean)

        if smooth and len(mean) >= 3:
            wl, po = _valid_savgol_params(len(mean), sg_window, sg_poly)
            mean_s = savgol_filter(mean, window_length=wl, polyorder=po)
            sd_s = savgol_filter(sd, window_length=wl, polyorder=po)
            sd_s = np.maximum(sd_s, 0.0)  # guard against tiny negatives from filtering
        else:
            mean_s, sd_s = mean, sd

        steps = np.arange(len(mean_s))
        line, = ax.plot(steps, mean_s, linewidth=2.6, label=optimizer)

        if show_band and x.shape[0] > 1:
            ax.fill_between(
                steps,
                mean_s - sd_s,
                mean_s + sd_s,
                alpha=band_alpha,
                color=line.get_color(),
                linewidth=0,
            )

    ax.set_title(f"{dataset_id} · {model_type} · {optimization_setup}")
    ax.set_xlabel("optimization steps")
    ax.set_ylabel("loss")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()




def plot_optimization_results_same_ds_per_optimizer_(model_type:str, dataset_id:str, testset_selection:str, optimization_setup:str, optimizers:list):
    results = aggregate_optimization_metrics(model_type, dataset_id, testset_selection, optimization_setup, optimizers)

    plt.figure()

    for optimizer, stats in results.items():
        x_mean = stats['X_LOSS_MEAN']
        #mean = x_mean[1] #.mean(axis=0)
        mean = x_mean.mean(axis=0)
        var = x_mean.var(axis=0)
        sd = x_mean.var(axis=0)

        steps = np.arange(len(mean))

        line, = plt.plot(steps, mean, label=optimizer)

        #lower = x_mean.min(axis=0)
        #upper = x_mean.max(axis=0)
        #plt.fill_between(steps, lower, upper, alpha=0.2)

        #plt.fill_between(steps, mean - sd, mean + sd, alpha=0.2)

    plt.title(f'{dataset_id} {model} {optimization_setup}')
    plt.xlabel('optimization steps')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return


def plot_all_trajectories_same_ds_per_optimizer(model_type: str, dataset_id: str, testset_selection: str, optimization_setup: str, optimizers: list, alpha: float = 0.15, lw: float = 1.0, max_lines_per_optimizer: int | None = None, seed: int = 0):
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots()

    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    opt_color = {}

    for optimizer in optimizers:
        try:
            x_gt, x_guess = retrieve_optimization_results_same_ds(
                model_type, dataset_id, testset_selection, optimization_setup, optimizer
            )
        except Exception:
            continue

        # loss per sample, per run, per step
        x_loss = np.abs(x_guess - x_gt).sum(axis=-1)  # (S, R, T)
        S, R, T = x_loss.shape
        trajs = x_loss.reshape(S * R, T)              # (S*R, T)

        # optionally downsample trajectories for readability/speed
        if max_lines_per_optimizer is not None and trajs.shape[0] > max_lines_per_optimizer:
            idx = rng.choice(trajs.shape[0], size=max_lines_per_optimizer, replace=False)
            trajs = trajs[idx]

        # assign one stable color per optimizer
        color = opt_color.setdefault(optimizer, next(color_cycle))

        steps = np.arange(T)
        for t in trajs:
            ax.plot(steps, t, color=color, alpha=alpha, linewidth=lw, label="_nolegend_")

        # legend handle (one entry per optimizer)
        ax.plot([], [], color=color, linewidth=2.5, label=optimizer)

    ax.set_title(f"{dataset_id} {model_type} {optimization_setup}")
    ax.set_xlabel("optimization steps")
    ax.set_ylabel("loss  (|x_guess - x_gt| sum over dims)")
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_first_sample_runs_same_ds_per_optimizer_(model_type: str, dataset_id: str, testset_selection: str, optimization_setup: str, optimizers: list, alpha: float = 0.2, lw: float = 1.0):
    fig, ax = plt.subplots()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    opt_color = {}

    for i, optimizer in enumerate(optimizers):
        try:
            x_gt, x_guess = retrieve_optimization_results_same_ds(
                model_type, dataset_id, testset_selection, optimization_setup, optimizer
            )
        except Exception:
            continue

        x_loss = np.abs(x_guess - x_gt).sum(axis=-1)   # (S, R, T)

        # ONLY first sample (sample index 0): keep all runs
        trajs = x_loss[0]                               # (R, T)
        R, T = trajs.shape

        color = opt_color.setdefault(optimizer, colors[i % len(colors)])
        steps = np.arange(T)

        for r in range(R):
            ax.plot(steps, trajs[r], color=color, alpha=alpha, linewidth=lw, label="_nolegend_")

        ax.plot([], [], color=color, linewidth=2.5, label=optimizer)

    ax.set_title(f"{dataset_id} {model_type} {optimization_setup}  (sample 0 only)")
    ax.set_xlabel("optimization steps")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    plt.show()



def plot_first_sample_runs_same_ds_per_optimizer(
    model_type: str,
    dataset_id: str,
    testset_selection: str,
    optimization_setup: str,
    optimizers: list,
    alpha_runs: float = 0.1,
    lw_runs: float = 1.0,
    lw_mean: float = 2.8,
    show_std_band: bool = False,
    std_band_alpha: float = 0.12,
):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    opt_color = {}

    for i, optimizer in enumerate(optimizers):
        try:
            x_gt, x_guess = retrieve_optimization_results_same_ds(
                model_type, dataset_id, testset_selection, optimization_setup, optimizer
            )
        except Exception:
            continue

        x_loss = np.abs(x_guess - x_gt).sum(axis=-1)  # (S, R, T)

        # ONLY first sample (sample index 0): keep all runs
        trajs = x_loss[0]  # (R, T)
        R, T = trajs.shape
        steps = np.arange(T)

        color = opt_color.setdefault(optimizer, colors[i % len(colors)])

        # Plot individual runs (transparent)
        ax.plot(steps, trajs.T, color=color, alpha=alpha_runs, linewidth=lw_runs)

        # Mean line (solid, clearly visible)
        mean_traj = trajs.mean(axis=0)
        ax.plot(steps, mean_traj, color=color, alpha=1.0, linewidth=lw_mean, label=optimizer)

        # Optional: std band around the mean
        if show_std_band and R > 1:
            std_traj = trajs.std(axis=0)
            ax.fill_between(
                steps,
                mean_traj - std_traj,
                mean_traj + std_traj,
                color=color,
                alpha=std_band_alpha,
                linewidth=0,
            )

    ax.set_title(f"{dataset_id} · {model_type} · {optimization_setup} (sample 0)")
    ax.set_xlabel("Optimization steps")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, ncol=1)
    fig.tight_layout()
    plt.show()
"""

GROUP_NAME = {1: "one parameter", 2: "two parameters", 3: "three parameters"}

def _setup_mask(setup_str: str) -> np.ndarray:
    mask = np.array(ast.literal_eval(setup_str), dtype=bool)
    return mask

def final_step_mae(x_gt: np.ndarray, x_guess: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    x_gt, x_guess: (S, R, T, D)
    returns: scalar MAE at final step, averaged over S and R, and over selected D
    """
    err = np.abs(x_guess - x_gt)  # (S,R,T,D)

    if mask is not None:
        if mask.ndim != 1:
            raise ValueError("mask must be 1D")
        # guard if mask longer than D
        D = err.shape[-1]
        mask = mask[:D]
        if mask.sum() == 0:
            return np.nan
        err = err[..., mask]  # (S,R,T,D_selected)

    mae_per_step = err.mean(axis=-1)     # (S,R,T)
    final_mae = mae_per_step[..., -1]    # (S,R)
    return float(final_mae.mean())       # scalar


def build_mae_table(model_type: str, dataset_ids: list[str], testset_selection: str, setups: list[str], optimizers: list[str]):
    """
    Returns a DataFrame with MultiIndex (group, optimizer) and columns dataset_ids.
    Each cell is: mean over all setups in that group of final-step MAE (averaged over S,R),
    computed only on the parameters selected by the setup mask.
    """
    rows = []

    for k, group_label in GROUP_NAME.items():
        setups_k = [s for s in setups if _setup_mask(s).sum() == k]

        for opt in optimizers:
            row = {"group": group_label, "optimizer": opt}

            for ds in dataset_ids:
                vals = []
                for setup in setups_k:
                    try:
                        x_gt, x_guess = retrieve_optimization_results_same_ds(
                            model_type=model_type,
                            dataset_id=ds,
                            testset_selection=testset_selection,
                            optimization_setup=setup,
                            optimizer=opt,
                        )
                        # your retrieve returns (S,R,T,D) after squeeze
                        mask = _setup_mask(setup)
                        vals.append(final_step_mae(x_gt, x_guess, mask=mask))
                    except Exception:
                        continue

                row[ds] = float(np.nanmean(vals)) if len(vals) else np.nan

            rows.append(row)

    df = pd.DataFrame(rows).set_index(["group", "optimizer"])
    return df


def build_mae_table_per_setup(
    model_type: str,
    dataset_ids: list[str],
    testset_selection: str,
    setups: list[str],
    optimizers: list[str],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    DataFrame index: (optimization_setup, optimizer)
    columns: dataset_ids
    values: final-step MAE for the parameters selected by the setup mask
    """
    rows = []

    for setup in setups:
        mask = _setup_mask(setup)

        for opt in optimizers:
            row = {"setup": setup, "optimizer": opt}

            for ds in dataset_ids:
                try:
                    x_gt, x_guess = retrieve_optimization_results_same_ds(
                        model_type=model_type,
                        dataset_id=ds,
                        testset_selection=testset_selection,
                        optimization_setup=setup,
                        optimizer=opt,
                    )
                    row[ds] = final_step_mae(x_gt, x_guess, mask=mask)
                except Exception as e:
                    row[ds] = np.nan
                    if verbose:
                        print(f"[missing] model={model_type}, ds={ds}, sel={testset_selection}, setup={setup}, opt={opt} -> {e}")

            rows.append(row)

    df = pd.DataFrame(rows).set_index(["setup", "optimizer"])
    return df



def build_mae_table_per_setup_with_sd(
    model_type: str,
    dataset_ids: list[str],
    testset_selection: str,
    setups: list[str],
    optimizers: list[str],
    ddof: int = 1,
    decimals: int = 3,
    format_cells: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
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

        for opt in optimizers:
            row = {"setup": setup, "optimizer": opt}

            for ds in dataset_ids:
                try:
                    x_gt, x_guess = retrieve_optimization_results_same_ds(
                        model_type=model_type,
                        dataset_id=ds,
                        testset_selection=testset_selection,
                        optimization_setup=setup,
                        optimizer=opt,
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
                        print(f"[missing] model={model_type}, ds={ds}, sel={testset_selection}, setup={setup}, opt={opt} -> {e}")

            rows.append(row)

    df = pd.DataFrame(rows).set_index(["setup", "optimizer"])
    return df


if __name__ == '__main__':
    model_types = ['ff', 'res', 'hres'] #, 'autotabpfn']
    dataset_ids = ['ds1', 'ds2', 'ds3', 'ds4', 'ds5', 'ds6', 'ds7', 'ds8']
    testset_selections = ['end'] # ['start', 'end', 'random']
    metrics = ['mae', 'mse', 'rmse', 'r2']
    optimizers = ['SGD', 'ASGD', 'SGD-nesterov','Adam', 'RMSprop']
    #optimizers = ['RMSprop']

    setups = ['[True, False, False]',
              '[False, True, False]',
              '[False, False, True]',

              '[True, True, False]',
              '[False, True, True]',
              '[True, False, True]',

              '[True, True, True]']

    #for model in model_types:

    model = 'res'


    print(f'====={model}=====')
    df = build_mae_table_per_setup_with_sd(model_type=model, dataset_ids=dataset_ids, testset_selection='end', setups=setups, optimizers=optimizers)
    with pd.option_context(
                "display.max_rows", None,
                "display.max_columns", None,
                "display.width", 200,
                "display.expand_frame_repr", False
        ):
            print(df)


    setups = ['[True, False, False]', '[False, True, True]', '[True, True, True]'] #, '[False, True, False]', '[False, True, True]', '[False, True, True]', '[True, True, True]', '[True, False, False]' , '[False, True, True]']
    dataset_ids = ['ds1', 'ds3', 'ds5']
    model_types = ['res', 'hres', 'ff']


    for setup in setups:  # [setups[1]]:
        for dataset_id in dataset_ids: # [dataset_ids[0]]:
            for model in model_types:
                plot_optimization_results_same_ds_per_optimizer(model, dataset_id, 'end', setup, optimizers, max_steps=700, headline='')



                #plot_all_trajectories_same_ds_per_optimizer(model, dataset_id, "end", setup, optimizers, alpha=0.12, lw=0.9, max_lines_per_optimizer=300)
                #plot_first_sample_runs_same_ds_per_optimizer(model, dataset_id, "end", setup, optimizers)

    #aggregate_optimization_metrics('ff', 'ds1', 'start', '[False, False, True]', ['Adam'])
    #results = retrieve_optimization_results_same_ds('ff', 'ds1', 'start', '[False, False, True]', 'Adam')

    """"""
    print('hurray')