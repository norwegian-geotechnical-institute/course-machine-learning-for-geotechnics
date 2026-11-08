"""Plotting functionality"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_predictions(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    labels: pd.Series | None = None,
    show_labels: bool = False,
    streamlit: bool = True,
) -> None:
    depth = df["Depth (m)"]
    qc = df["qc (MPa)"]
    fs = df["fs (kPa)"]

    plt.rcParams.update(
        {
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 18,
        }
    )

    fig, axs = plt.subplots(
        1, 4 if labels is not None else 3, figsize=(20, 10), sharey=True
    )

    # Plot qc (MPa) vs Depth
    axs[0].plot(qc, depth, color="b")
    axs[0].set_xlabel("qc (MPa)")
    axs[0].set_ylabel("Depth (m)")
    axs[0].invert_yaxis()
    axs[0].set_title("qc (MPa) vs Depth")

    # Plot fs (kPa) vs Depth
    axs[1].plot(fs, depth, color="g")
    axs[1].set_xlabel("fs (kPa)")
    axs[1].set_title("fs (kPa) vs Depth")

    # Plot predicted labels vs Depth
    axs[2].plot(y_pred, depth, color="r")
    axs[2].set_xlabel("Predicted Classes")
    axs[2].set_title("Predicted vs Depth")

    # Plot true labels vs Depth (only if labels are provided)
    if labels is not None and show_labels:
        axs[3].plot(labels, depth, color="orange")
        axs[3].set_xlabel("True Classes")
        axs[3].set_title("True Labels vs Depth")

    plt.tight_layout()
    if streamlit:
        st.pyplot(fig)
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: list[Any] | pd.Series,
    y_pred: list[Any] | pd.Series,
    class_mapping: dict[int, str],
    normalize: str = "true",
    add_black_lines: bool = True,
) -> plt.Figure:
    """
    Plots a confusion matrix with options for customization.

    Parameters:
    y_true (list[Any]): True labels.
    y_pred (list[Any]): Predicted labels.
    class_mapping (dict[int, str]): Mapping of class numbers to class names.
    normalize (str, optional): Normalization option for confusion matrix. Defaults to 'true'.
    add_black_lines (bool, optional): Whether to add black grid lines around each square. Defaults to True.

    Example class_mapping:

    soil_classification = {
    1: "gravel",
    4: "sand to gravel",
    3: "coarse grained organic soils",
    5: "sand",
    2: "fine grained organic soils",
    6: "silt to fine sand",
    7: "clay to silt"
    }

    """
    # Determine the set of labels present in the data
    y_true_series = pd.Series(y_true)
    y_pred_series = pd.Series(y_pred)
    present_labels = pd.unique(
        pd.concat([y_true_series, y_pred_series], ignore_index=True)
    )

    # Filter class mapping to only present labels; if mapping missing new labels, fallback to stringified numeric labels
    present_labels_series = pd.Series(present_labels)
    present_labels_numeric = pd.to_numeric(present_labels_series, errors="coerce")
    # Build ordered lists for confusion matrix display, handling both numeric and non-numeric labels
    class_label_numbers: list[Any] = []
    class_labels: list[str] = []
    for orig_lbl, num_lbl in zip(
        present_labels_series, present_labels_numeric, strict=True
    ):
        if pd.notna(num_lbl):
            lbl = int(num_lbl)
            class_label_numbers.append(lbl)
            class_labels.append(class_mapping.get(lbl, str(lbl)))
        else:
            class_label_numbers.append(orig_lbl)
            class_labels.append(str(orig_lbl))
    # Generate the confusion matrix with labels in the desired order
    conf_matrix = confusion_matrix(
        y_true, y_pred, labels=class_label_numbers, normalize=normalize
    )

    # Round the values in the confusion matrix to a maximum of 3 digits behind the comma
    conf_matrix = np.round(conf_matrix, 3)

    # Visualise the confusion matrix with an optional thin black line around each square
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=class_labels
    )
    disp.plot(cmap="Blues", ax=ax, colorbar=False)

    # Toggle black grid lines on or off
    if add_black_lines:
        add_black_grid_lines(ax, conf_matrix)

    # Ensure labels align correctly with ticks
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_yticklabels(class_labels)

    for text in disp.text_.ravel():
        text.set_fontsize(12)  # Adjust the size to your liking

    plt.title("Confusion Matrix (recall for each class on the diagonal)")
    plt.tight_layout()
    return fig


def add_black_grid_lines(ax: plt.Axes, conf_matrix: np.ndarray) -> None:
    """
    Adds black grid lines around each square in the confusion matrix.

    Parameters:
    ax (plt.Axes): The axes object of the plot.
    conf_matrix (np.ndarray): The confusion matrix data.
    """
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_edgecolor("black")
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.set_xticks(np.arange(conf_matrix.shape[1]) + 0.5, minor=True)
    ax.set_yticks(np.arange(conf_matrix.shape[0]) + 0.5, minor=True)
    ax.tick_params(which="minor", size=0)


def plot_qc_boxplot_by_oberhollenzer_classes(
    csv_path: str | Path = "data/model_ready/cpt_dataset_total.csv",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Create a boxplot of qc for each Oberhollenzer class (8 boxes expected).

    Parameters:
    - csv_path: Path to the model-ready dataset CSV containing columns for qc and Oberhollenzer_classes.
    - save_path: Optional output image path. If not provided, saves under plots/EDA/qc_boxplot_by_oberhollenzer_classes.png.
    - show: If True, display the plot window (useful outside notebooks/Streamlit).

    Returns:
    - The matplotlib Figure object for further manipulation or testing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    # Read data
    df = pd.read_csv(csv_path, low_memory=False)

    # Detect qc column (dataset commonly uses 'qc (MPa)')
    qc_candidates = [
        "qc (MPa)",
        "qc",
        "qc_MPa",
        "qc(MPa)",
    ]
    qc_col = next((c for c in qc_candidates if c in df.columns), None)
    if qc_col is None:
        raise KeyError(
            "Could not find a qc column in the dataset. Looked for: "
            + ", ".join(qc_candidates)
        )

    class_col = "Oberhollenzer_classes"
    if class_col not in df.columns:
        raise KeyError(
            f"Could not find '{class_col}' column in the dataset. Available columns: {list(df.columns)[:20]}..."
        )

    # Clean and prepare
    plot_df = df[[qc_col, class_col]].dropna()
    # Ensure classes are integers (values are typically 0..7 or 1..7)
    plot_df[class_col] = pd.to_numeric(plot_df[class_col], errors="coerce").astype(
        "Int64"
    )
    plot_df = plot_df.dropna(subset=[class_col])
    plot_df[class_col] = plot_df[class_col].astype(int)

    # Determine class order (expect 8 classes 0..7). Use present classes if fewer to avoid errors.
    expected_order = list(range(8))
    present_classes = sorted(plot_df[class_col].unique().tolist())
    if set(expected_order).issubset(set(present_classes)):
        class_order = expected_order
    else:
        class_order = present_classes  # Fall back gracefully if not all 8 are present

    data = [plot_df.loc[plot_df[class_col] == k, qc_col].values for k in class_order]

    # Configure plot appearance
    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(data, labels=[str(k) for k in class_order], showfliers=False)
    ax.set_xlabel("Oberhollenzer class")
    ax.set_ylabel("qc (MPa)" if qc_col == "qc (MPa)" else qc_col)
    title_suffix = (
        " (all 8 classes)"
        if class_order == expected_order
        else f" (classes: {class_order})"
    )
    ax.set_title(f"qc distribution by Oberhollenzer class{title_suffix}")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()

    # Save if requested or default location
    if save_path is None:
        save_path = Path("plots/EDA/qc_boxplot_by_oberhollenzer_classes.png")
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)

    if show:
        plt.show()

    return fig
