import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import itertools
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.lines import Line2D


def plot_results(main_folder, plot_std=False):
    """Plots the experiment results averaged over the seeds."""

    experiments = ['default', 'small', 'large', 'transfer_small', 'transfer_large']
    seeds = ['0', '1', '2', '3', '4']
    metric = 'score'
    output_file = os.path.join(main_folder, 'results.pdf')

    # Collect data
    experiment_results = {}

    for exp in experiments:
        seed_dfs = []
        for seed in seeds:
            file_path = os.path.join(main_folder, exp, seed, 'val_log.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0)
                # keep only the scores created after transfer
                if df.shape[0] > len(seeds) + 1:
                    df = df.tail(len(seeds) + 1)
                seed_dfs.append(df)
            else:
                print(f"Warning: Missing file {file_path}")
        if seed_dfs:
            # Stack into 3D array for mean/std
            combined = pd.concat(seed_dfs, axis=0, keys=range(len(seed_dfs)))
            mean_df = combined.groupby(level=1).mean()
            std_df = combined.groupby(level=1).std()
            experiment_results[exp] = {'mean': mean_df, 'std': std_df}
        else:
            print(f"Warning: No valid seed logs found for {exp}")

    # Plotting
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
    except Exception as e:
    # switch to a headless backend (https://stackoverflow.com/questions/4706451/how-to-save-a-figure-remotely-with-pylab)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))

    for exp_name, data in experiment_results.items():
        if metric in data['mean'].columns:
            steps = data['mean'].index
            mean_values = data['mean'][metric]
            plt.plot(steps, mean_values, label=exp_name)
            if plot_std:
                std_values = data['std'][metric]
                plt.fill_between(steps, mean_values - std_values, mean_values + std_values, alpha=0.2)

    # plt.title(f'Metric: {metric}')
    plt.xlabel("Number of samples")
    plt.ylabel("Validation score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save to PDF
    with PdfPages(output_file) as pdf:
        pdf.savefig()
        print(f"Saved plot to {output_file}")


def smooth_curve(x, y, num_points=200):
    """Smooth a curve with cubic spline interpolation."""
    if len(x) < 4:
        return x, y
    x_new = np.linspace(x.min(), x.max(), num_points)
    spline = make_interp_spline(x, y, k=3)
    y_new = spline(x_new)
    return x_new, y_new


def rolling_average(y, window=5):
    """Return a simple rolling average (same length as input)."""
    return pd.Series(y).rolling(window=window, min_periods=1, center=True).mean().values


def plot_merged_transfer(
    folder_dimensional,
    folder_dimensionless,
    output_path,
    plot_std=True,
    plot_seeds=False,
    smooth=True,
    smooth_window=5,
):
    """Plot transfer comparison between dimensional and dimensionless experiments."""

    def load_results(folder):
        csv_path = os.path.join(folder, "results.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing results.csv in {folder}")
        return pd.read_csv(csv_path)

    df_dim = load_results(folder_dimensional)
    df_dimless = load_results(folder_dimensionless)

    step_col = df_dim['step']
    default_series = df_dim['default'].dropna()
    default_steps = step_col.loc[default_series.index]
    transition_point = default_steps.max()

    plt.rcParams.update({
        'font.size': 14,              # base font
        'axes.titlesize': 16,
        'axes.labelsize': 18,         # axis labels
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'legend.title_fontsize': 14,
    })
    plt.rcParams['pdf.fonttype'] = 42  # Ensure fonts are embedded properly in PDF

    plt.figure(figsize=(10, 6))
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    experiments = ['default', 'transfer_small', 'transfer_large']

    color_map = {}

    for exp_name in experiments:
        color = next(color_cycle)
        color_map[exp_name] = color

        # Clean display names
        if exp_name == "default":
            label_pretty = "default"
        elif exp_name == "transfer_small":
            label_pretty = "small"
        elif exp_name == "transfer_large":
            label_pretty = "large"
        else:
            label_pretty = exp_name

        # ===== Dimensionless (solid) =====
        mean_dimless = df_dimless[exp_name].dropna()
        steps_dimless = step_col.loc[mean_dimless.index]
        step_shift = transition_point - steps_dimless.min() if "transfer" in exp_name else 0.0

        x_vals = steps_dimless.values + step_shift
        y_vals = mean_dimless.values

        if smooth is True:
            x_vals, y_vals = smooth_curve(x_vals, y_vals)
        elif smooth == "rolling":
            y_vals = rolling_average(y_vals, window=smooth_window)

        plt.plot(x_vals, y_vals, color=color, linewidth=2.5, linestyle='-')

        if plot_std and exp_name + "_std" in df_dimless.columns:
            std_vals = df_dimless[exp_name + "_std"].dropna().values
            if smooth is True:
                _, std_vals = smooth_curve(steps_dimless.values + step_shift, std_vals)
            elif smooth == "rolling":
                std_vals = rolling_average(std_vals, window=smooth_window)
            plt.fill_between(x_vals, y_vals - std_vals, y_vals + std_vals, color=color, alpha=0.15)
            plt.plot(x_vals, y_vals - std_vals, color=color, linestyle='-', linewidth=0.8)
            plt.plot(x_vals, y_vals + std_vals, color=color, linestyle='-', linewidth=0.8)

        # ===== Dimensional (dashed) =====
        mean_dim = df_dim[exp_name].dropna()
        steps_dim = step_col.loc[mean_dim.index]
        step_shift_d = transition_point - steps_dim.min() if "transfer" in exp_name else 0.0

        x_vals_d = steps_dim.values + step_shift_d
        y_vals_d = mean_dim.values

        if smooth is True:
            x_vals_d, y_vals_d = smooth_curve(x_vals_d, y_vals_d)
        elif smooth == "rolling":
            y_vals_d = rolling_average(y_vals_d, window=smooth_window)

        plt.plot(x_vals_d, y_vals_d, color=color, linestyle='--', linewidth=2.5)

        if plot_std and exp_name + "_std" in df_dim.columns:
            std_vals_d = df_dim[exp_name + "_std"].dropna().values
            if smooth is True:
                _, std_vals_d = smooth_curve(steps_dim.values + step_shift_d, std_vals_d)
            elif smooth == "rolling":
                std_vals_d = rolling_average(std_vals_d, window=smooth_window)
            plt.fill_between(x_vals_d, y_vals_d - std_vals_d, y_vals_d + std_vals_d, color=color, alpha=0.15)
            plt.plot(x_vals_d, y_vals_d - std_vals_d, color=color, linestyle='--', linewidth=0.8)
            plt.plot(x_vals_d, y_vals_d + std_vals_d, color=color, linestyle='--', linewidth=0.8)

    # Vertical line at transfer start
    plt.axvline(x=transition_point, color='black', linestyle='--', label='transfer start')

    # === Grouped Legends ===
    exp_handles = [
        Line2D([0], [0], color=color_map['default'], lw=2.5, label='default'),
        Line2D([0], [0], color=color_map['transfer_small'], lw=2.5, label='small'),
        Line2D([0], [0], color=color_map['transfer_large'], lw=2.5, label='large'),
    ]
    system_handles = [
        Line2D([0], [0], color='k', lw=2.5, linestyle='-', label='dimensionless'),
        Line2D([0], [0], color='k', lw=2.5, linestyle='--', label='dimensional'),
    ]

    # Place legends close together in the upper-left
    first_legend = plt.legend(
        handles=exp_handles,
        title="System scale",
        loc='upper left',
        bbox_to_anchor=(0.02, 1.0),
        frameon=True
    )
    plt.gca().add_artist(first_legend)

    # Move the second legend right next to the first
    plt.legend(
        handles=system_handles,
        title="Formulation",
        loc='upper left',
        bbox_to_anchor=(0.2, 1.0),
        frameon=True
    )

    # === Final formatting ===
    plt.ylim((0.0, 20.0))
    plt.xlabel("Number of samples")
    plt.ylabel("Validation score")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path)
    print(f"Saved merged transfer plot to {output_path}")


if __name__ == "__main__":
    folder_dim = "/home/josip/dimensionless-mpcrl/output/cart_pole/learning_progress/learning_dimensional_20250827110008"
    folder_dimless = "/home/josip/dimensionless-mpcrl/output/cart_pole/learning_progress/learning_dimensionless_20250826105216"

    # # Option 1: spline smoothing (default)
    # plot_merged_transfer(
    #     folder_dimensional=folder_dim,
    #     folder_dimensionless=folder_dimless,
    #     output_path="merged_transfer_spline.pdf",
    #     smooth=True
    # )

    # # Option 2: rolling average
    # plot_merged_transfer(
    #     folder_dimensional=folder_dim,
    #     folder_dimensionless=folder_dimless,
    #     output_path="merged_transfer_rolling.pdf",
    #     smooth="rolling",
    #     smooth_window=3
    # )

    # Option 3: raw curves
    plot_merged_transfer(
        folder_dimensional=folder_dim,
        folder_dimensionless=folder_dimless,
        output_path=os.path.join(os.path.dirname(folder_dim), "merged_transfer_plot.pdf"),
        smooth=False
    )