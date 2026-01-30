import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import itertools
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.lines import Line2D


def plot_results(main_folder, plot_std, plot_seeds):
    """Plots the experiment results averaged over the seeds."""

    # assume the same names as in the run script
    experiments = ['default', 'small', 'large', 'transfer_small', 'transfer_large']
    # detect the number of seeds automatically
    seeds = sorted([
        d for d in os.listdir(os.path.join(main_folder, experiments[0]))
        if d.isdigit()
    ], key=int)
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
                reset_positions = df.index.to_series().eq(0).to_numpy().nonzero()[0]
                if len(reset_positions) > 1:
                    df = df.iloc[reset_positions[1]:]
                seed_dfs.append(df)
            else:
                print(f"Warning: Missing file {file_path}")
        if seed_dfs:
            # Stack into 3D array for mean/std
            combined = pd.concat(seed_dfs, axis=0, keys=range(len(seed_dfs)))
            mean_df = combined.groupby(level=1).mean()
            std_df = combined.groupby(level=1).std()
            experiment_results[exp] = {
                'mean': mean_df,
                'std': std_df,
                'seeds': seed_dfs  # list of DataFrames for each seed
            }
        else:
            print(f"Warning: No valid seed logs found for {exp}")

    # Save wide-format CSV: one row per step, one column per experiment (mean), optionally std
    results_csv_path = os.path.join(main_folder, 'results.csv')

    combined_df = pd.DataFrame()

    for exp_name, data in experiment_results.items():
        mean_series = data['mean'][metric].rename(exp_name)
        combined_df = pd.concat([combined_df, mean_series], axis=1)

    if plot_std:
        for exp_name, data in experiment_results.items():
            std_series = data['std'][metric].rename(exp_name + "_std")
            combined_df = pd.concat([combined_df, std_series], axis=1)

    combined_df.index.name = "step"
    combined_df.to_csv(results_csv_path)
    print(f"Saved the averaged data to {results_csv_path}")

    # Plotting
    try:
        plt.figure(figsize=(8, 6))
    except Exception as e:
    # switch to a headless backend (https://stackoverflow.com/questions/4706451/how-to-save-a-figure-remotely-with-pylab)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
    
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    for exp_name, data in experiment_results.items():
        if metric in data['mean'].columns:
            steps = data['mean'].index
            mean_values = data['mean'][metric]
            
            # Get the next color in the cycle
            color = next(color_cycle)
            
            # Plot mean
            plt.plot(steps, mean_values, label=exp_name, color=color)

            # Plot individual seeds with lower alpha
            if plot_seeds:
                for seed_df in data['seeds']:
                    if metric in seed_df.columns:
                        plt.plot(seed_df.index, seed_df[metric], color=color, alpha=0.2, linewidth=1)

            # Plot std band
            if plot_std:
                std_values = data['std'][metric]
                plt.fill_between(steps, mean_values - std_values, mean_values + std_values, alpha=0.1, color=color)

    # plt.title(f'Metric: {metric}')
    plt.ylim((0.0, 20.0))
    plt.xlabel("Number of samples")
    plt.ylabel("Validation score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save to PDF
    with PdfPages(output_file) as pdf:
        pdf.savefig()
        print(f"Saved the classic plot to {output_file}")

    plot_transfer(
        main_folder=main_folder,
        output_path=os.path.join(main_folder, "transfer_plot.pdf"),
        plot_std=plot_std,
        plot_seeds=plot_seeds
    )


def plot_transfer(main_folder, output_path, plot_std, plot_seeds):
    csv_path = os.path.join(main_folder, 'results.csv')
    df = pd.read_csv(csv_path)

    # Determine step interval (assumes constant interval)
    step_col = df['step']
    step_interval = step_col.diff().dropna().iloc[0]

    # Get default data
    default_series = df['default'].dropna()
    default_steps = step_col.loc[default_series.index]
    transition_point = default_steps.max()

    # Start figure
    plt.figure(figsize=(10, 6))
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # Plot default (mean)
    color_default = next(color_cycle)
    plt.plot(default_steps, default_series, label='default', color=color_default)

    # Plot std for default
    if plot_std and 'default_std' in df.columns:
        default_std = df['default_std'].dropna()
        std_steps = step_col.loc[default_std.index]
        plt.fill_between(
            std_steps,
            default_series - default_std,
            default_series + default_std,
            color=color_default,
            alpha=0.2,
        )

    # Plot seeds for default
    seeds = sorted([
        d for d in os.listdir(os.path.join(main_folder, 'default'))
        if d.isdigit()
    ], key=int)
    if plot_seeds:
        for seed in seeds:
            file_path = os.path.join(main_folder, 'default', seed, 'val_log.csv')
            if os.path.exists(file_path):
                seed_df = pd.read_csv(file_path, index_col=0)
                if 'score' in seed_df.columns:
                    plt.plot(
                        seed_df.index,
                        seed_df['score'],
                        color=color_default,
                        alpha=0.3,
                        linewidth=1
                    )
            else:
                print(f"Warning: Missing file {file_path}")


    # Dashed line at transition point
    plt.axvline(x=transition_point, color='black', linestyle='--', label='transfer start')

    # Plot transfer experiments
    for transfer_key in ['transfer_small', 'transfer_large']:
        color = next(color_cycle)

        # Plot mean from results.csv
        series = df[transfer_key].dropna()
        steps = step_col.loc[series.index]
        step_shift = transition_point - steps.min()
        aligned_steps = steps + step_shift
        plt.plot(aligned_steps, series, label=transfer_key.removeprefix("transfer_"), color=color)

        # Optional: plot std (if plot_std is True)
        if plot_std:
            std_key = transfer_key + "_std"
            if std_key in df.columns:
                std_series = df[std_key].dropna()
                std_steps = step_col.loc[std_series.index]
                aligned_std_steps = std_steps + step_shift
                plt.fill_between(
                    aligned_std_steps,
                    series - std_series,
                    series + std_series,
                    color=color,
                    alpha=0.2,
                )

        # Optional: plot individual seeds (if plot_seeds is True)
        if plot_seeds:
            for seed in seeds:
                file_path = os.path.join(main_folder, transfer_key, seed, 'val_log.csv')
                if os.path.exists(file_path):
                    seed_df = pd.read_csv(file_path, index_col=0)
                    if seed_df.shape[0] > 6:  # keep only post-transfer steps
                        seed_df = seed_df.tail(6)
                    if 'score' in seed_df.columns:
                        plt.plot(
                            seed_df.index + step_shift,  # shift x-axis
                            seed_df['score'],
                            color=color,
                            alpha=0.3,
                            linewidth=1
                        )
                else:
                    print(f"Warning: Missing file {file_path}")

    # Labels, legend, etc.
    plt.ylim((0.0, 20.0))
    plt.xlabel("Number of samples")
    plt.ylabel("Validation score")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path)
    print(f"Saved the transfer plot to {output_path}")


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
    print(f"Saved the merged transfer plot to {output_path}")


if __name__ == "__main__":
    # change the paths below to point to your experiment folders
    folder_dim = ""
    folder_dimless = ""

    print("Plotting results for the dimensional formulation...")
    plot_results(main_folder=folder_dim, plot_std=True, plot_seeds=False)
    print("-"*50)

    print("Plotting results for the dimensionless formulation...")
    plot_results(main_folder=folder_dimless, plot_std=True, plot_seeds=False)
    print("-"*50)    

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
        output_path=os.path.join(os.path.dirname(os.path.dirname(folder_dim)), "merged_transfer_plot.pdf"),
        plot_std=False,
        smooth=False
    )