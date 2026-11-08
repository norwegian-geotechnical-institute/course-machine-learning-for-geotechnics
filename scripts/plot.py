from pathlib import Path

from cpt_to_soiltype.plotting import (
    plot_qc_boxplot_by_oberhollenzer_classes,
)


def main() -> None:
    # Ensure output directory exists
    out_dir = Path("plots/EDA")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "qc_boxplot_by_oberhollenzer_classes.png"

    # Create and save the figure
    plot_qc_boxplot_by_oberhollenzer_classes(
        csv_path="data/model_ready/cpt_dataset_total.csv",
        save_path=out_file,
        show=False,
    )


if __name__ == "__main__":
    main()
