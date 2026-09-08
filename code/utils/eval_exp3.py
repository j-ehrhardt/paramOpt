import os
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_RESULTS_ROOT = REPO_ROOT / 'results' / 'paper' / 'exp3'

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


def _load_result_arrays(file_path: Path):
    with file_path.open('r') as f:
        results = json.load(f)

    x_gt_list = results['X_GROUND_TRUTH']
    x_guess_list = results['X_GUESS']
    max_L = max(max(len(s[0]) for s in x_gt_list), max(len(s[0]) for s in x_guess_list))
    x_gt = pad_samples(x_gt_list, max_L).squeeze(axis=3)
    x_guess = pad_samples(x_guess_list, max_L).squeeze(axis=3)
    return x_gt, x_guess


def retrieve_optimization_results_same_ds(model_type: str, dataset_id: str,
                                          testset_selection: str,
                                          optimization_setup: str, paradigm: str):
    """Load all saved seed runs for one paper-evaluation configuration.

    The returned run axis is the seed axis (or seed × explicit independent
    restart), never a chain of successive perturbations.
    """
    directory = PAPER_RESULTS_ROOT / dataset_id / model_type / testset_selection
    if paradigm == 'paramopt':
        suffix = f'_on_{dataset_id}_{optimization_setup}_paramopt_RMSprop.json'
    else:
        suffix = f'_on_{optimization_setup}_{paradigm}.json'
    prefix = f'{model_type}_{dataset_id}_{testset_selection}_'
    files = sorted(path for path in directory.glob('*.json')
                   if path.name.startswith(prefix) and path.name.endswith(suffix))
    if not files:
        raise FileNotFoundError(f'No result files for {model_type}, {dataset_id}, {optimization_setup}, {paradigm}.')

    arrays = [_load_result_arrays(path) for path in files]
    n_samples = arrays[0][0].shape[0]
    if any(x_gt.shape[0] != n_samples for x_gt, _ in arrays):
        raise ValueError('Seed runs have different test-set sizes and cannot be aggregated.')

    max_steps = max(x_guess.shape[2] for _, x_guess in arrays)
    padded = []
    for x_gt, x_guess in arrays:
        if x_guess.shape[2] < max_steps:
            pad_width = max_steps - x_guess.shape[2]
            x_guess = np.concatenate((x_guess, np.repeat(x_guess[:, :, -1:, :], pad_width, axis=2)), axis=2)
            x_gt = np.concatenate((x_gt, np.repeat(x_gt[:, :, -1:, :], pad_width, axis=2)), axis=2)
        padded.append((x_gt, x_guess))

    return (
        np.concatenate([x_gt for x_gt, _ in padded], axis=1),
        np.concatenate([x_guess for _, x_guess in padded], axis=1),
    )



def build_mae_table_per_setup_with_sd(model_type: str, dataset_ids: list[str], testset_selection: str, setups: list[str], paradigms: list[str], ddof: int = 1, decimals: int = 1, format_cells: bool = True, verbose: bool = False):
    """
    Rows: (setup, optimizer)
    Cols: ds1..ds6

    Each cell summarizes final-step MAE across independent seed runs:
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
        return final_mae.mean(axis=0)      # (R,), one test-set MAE per seed

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
    paradigms,
    max_steps: int | None = None,     # NEW: clip number of steps (e.g., 200)
    headline: str | None = None,      # NEW: custom title
    smooth: bool = False,
    sg_window: int = 21,
    sg_poly: int = 3,
    show_band: bool = False,
    band_alpha: float = 0.12,
    legend_alpha: float = 0.85,       # legend box transparency
):
    mask = np.array(ast.literal_eval(optimization_setup), dtype=bool)

    fig, ax = plt.subplots(figsize=(6, 4))

    for par in paradigms:
        try:
            x_gt, x_guess = retrieve_optimization_results_same_ds(
                model_type, dataset_id, testset_selection, optimization_setup, par
            )
        except Exception:
            continue

        # x_gt/x_guess: (S,R,T,D)
        err = np.abs(x_guess - x_gt)
        D = err.shape[-1]
        m = mask[:D]
        err = err[..., m]                        # (S,R,T,D_sel)

        mae = err.mean(axis=-1)                  # (S,R,T)
        mae_per_seed = mae.mean(axis=0)          # (R,T)

        mean = mae_per_seed.mean(axis=0)
        sd   = mae_per_seed.std(axis=0, ddof=1) if mae_per_seed.shape[0] > 1 else np.zeros_like(mean)

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
        line, = ax.plot(steps, mean, linewidth=2.6, label=par)

        if show_band and mae_per_seed.shape[0] > 1:
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



def plot_optimization_results_same_ds_per_run(
    model_type: str,
    dataset_id: str,
    testset_selection: str,
    optimization_setup: str,
    paradigms: list[str],
    max_steps: int | None = None,
    headline: str | None = None,
    smooth: bool = False,
    sg_window: int = 21,
    sg_poly: int = 3,
    run_alpha: float = 0.25,
    run_linewidth: float = 1.2,
    show_mean: bool = True,
    mean_linewidth: float = 2.6,
    legend_alpha: float = 0.85,
    aggregate_over_samples: bool = True,  # True: one curve per run (avg over samples). False: one curve per (sample,run).
):
    """
    Plots trajectories for each optimization run (seed) instead of the averaged (S*R) trajectory.

    Data shapes from retrieve_optimization_results_same_ds:
      x_gt, x_guess: (S, R, T, D)

    MAE per step is computed over selected parameter dims (mask), then:
      - aggregate_over_samples=True: curves are mean over samples => (R, T)  (one curve per run)
      - aggregate_over_samples=False: curves are (S*R, T) (one curve per sample-run; can be many!)
    """
    mask = np.array(ast.literal_eval(optimization_setup), dtype=bool)
    paradigm_dict = {
        'paramopt': 'paramOpt',
        'beam_search': 'beam search',
        'genetic_algorithm': 'genetic algorithm',
    }

    fig, ax = plt.subplots(figsize=(6, 4))

    for par in paradigms:
        try:
            x_gt, x_guess = retrieve_optimization_results_same_ds(
                model_type, dataset_id, testset_selection, optimization_setup, par
            )
        except Exception:
            continue

        err = np.abs(x_guess - x_gt)  # (S,R,T,D)
        D = err.shape[-1]
        m = mask[:D]
        if m.sum() == 0:
            continue

        err = err[..., m]                  # (S,R,T,D_sel)
        mae = err.mean(axis=-1)            # (S,R,T)

        if aggregate_over_samples:
            traj = mae.mean(axis=0)        # (R,T) one curve per run
        else:
            traj = mae.reshape(-1, mae.shape[-1])  # (S*R,T) one curve per sample-run

        # clip steps
        if max_steps is not None:
            Tclip = min(int(max_steps), traj.shape[1])
            traj = traj[:, :Tclip]

        # smoothing (optional)
        if smooth:
            if savgol_filter is None:
                raise ImportError("scipy is required for smooth=True (savgol_filter not available).")
            if traj.shape[1] >= 3:
                wl, po = _valid_savgol_params(traj.shape[1], sg_window, sg_poly)
                traj = np.vstack([savgol_filter(traj[i], window_length=wl, polyorder=po) for i in range(traj.shape[0])])

        steps = np.arange(traj.shape[1])

        # use one color per paradigm (consistent for all its runs)
        color = ax._get_lines.get_next_color()

        # plot all runs (thin + transparent)
        for i in range(traj.shape[0]):
            ax.plot(steps, traj[i], color=color, alpha=run_alpha, linewidth=run_linewidth)

        # optional mean line (thicker)
        if show_mean:
            mean = traj.mean(axis=0)
            ax.plot(steps, mean, color=color, linewidth=mean_linewidth, label=paradigm_dict[par])
        else:
            # still put a clean legend entry
            ax.plot([], [], color=color, linewidth=mean_linewidth, label=paradigm_dict[par])

    if headline is None:
        headline = f"{dataset_id} · {model_type} · {optimization_setup}"
    ax.set_title(headline)
    ax.set_xlabel("optimization steps")
    ax.set_ylabel("MAE")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=legend_alpha,
        borderpad=0.6,
    )

    fig.tight_layout()
    fig.savefig(f"{headline}_4.pdf", bbox_inches="tight")
    plt.show()



if __name__ == '__main__':
    dataset_ids = ['ds1', 'ds2', 'ds3', 'ds4', 'ds5', 'ds6']
    setups = ['[True, False, False]', '[False, True, False]', '[False, True, True]', '[True, True, True]']
    paradigms = ['paramopt', 'beam_search', 'genetic_algorithm']
    table = build_mae_table_per_setup_with_sd(
        model_type='res',
        dataset_ids=dataset_ids,
        testset_selection='end',
        setups=setups,
        paradigms=paradigms,
    )
    print(table)
