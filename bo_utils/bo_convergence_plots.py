import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_convergence_from_csv(
    csv_path,
    metric="mse",
    group_by=["acquisition_type", "num_init"],
    label_fmt="{acquisition_type}, init={num_init}",
    title="Convergence Plot",
    figsize=(10, 6),
    save_path=None,
    y_max_true=None,
):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    # check the separator in the CSV file

    df = pd.read_csv(csv_path, sep=';')

    fig, ax = plt.subplots(figsize=figsize)

    grouped = df.groupby(group_by)
    for group_key, group_df in grouped:
        label = label_fmt.format(**dict(zip(group_by, group_key)))

        summary = (
            group_df.groupby("iteration")[metric]
            .agg([
                ("median", "median"),
                ("q25", lambda x: np.percentile(x, 25)),
                ("q75", lambda x: np.percentile(x, 75)),
            ])
            .reset_index()
        )
        if y_max_true is not None:
            ax.axhline(y=y_max_true, color='red', linestyle='--', label='True Max Value')
        ax.plot(summary["iteration"], summary["median"], label=label)
        ax.fill_between(summary["iteration"], summary["q25"], summary["q75"], alpha=0.3)

    ax.set_xlabel("BO Iteration")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metric_boxplot(
    csv_path,
    metric="mse",
    x_axis="num_init",
    hue="acquisition_type",
    title=None,
    save_path=None,
    figsize=(10, 6),
    iteration_cutoff=None  # Optional: nur Werte ab letzter Iteration betrachten
):
    df = pd.read_csv(csv_path)

    if iteration_cutoff is not None:
        df = df[df["iteration"] >= iteration_cutoff]

    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x=x_axis, y=metric, hue=hue)

    plt.title(title or f"Boxplot of {metric}")
    plt.ylabel(metric.upper())
    plt.xlabel(x_axis)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()