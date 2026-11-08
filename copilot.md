# Copilot Instructions for CPT-to-Soiltype ML Project

## Project Overview
This is a geotechnical machine learning project that predicts soil types (1-7: gravel to clay) from Cone Penetration Test (CPT) data using XGBoost. It's designed as an educational end-to-end ML pipeline for the Norwegian Geotechnical Institute.

## Architecture & Key Components

### Package Structure
- **`src/cpt_to_soiltype/`**: Core Python package with reusable ML functions
- **`scripts/`**: Entry point executables for training, preprocessing, prediction
- **`notebooks/`**: Educational Jupyter notebooks demonstrating concepts
- **`data/`**: Three-tier data structure (raw → intermediate → model_ready)

### Configuration System (Hydra)
All scripts use Hydra for configuration management with YAML files in `scripts/config/`:
- `main.yaml`: Main config with defaults pointing to model sub-configs
- `model/*.yaml`: Model-specific parameters (xgboost_native.yaml, knn.yaml, etc.)
- Configuration is validated via Pydantic schemas in `schema_config.py`

## Development Workflows

### Running Scripts
Always use `uv run` prefix for consistency:
```bash
uv run python scripts/train.py                    # Default XGBoost model
uv run python scripts/train.py model=knn         # Override model type
uv run python scripts/preprocess.py              # Data preprocessing
uv run python scripts/optimise_hyperparameters.py  # Optuna optimization (XGBoost only)
```

### MLflow Integration
- Experiments are tracked in `./experiments/` directory
- Start MLflow UI: `cd experiments && mlflow ui`
- Model artifacts and metrics automatically logged via `log_mlflow_metrics_and_model()`

### Streamlit Application
- Main demo app: `streamlit run Main.py --server.port 8503`
- Uses port 8503 to avoid MLflow conflicts
- Loads pre-trained XGBoost model from `models/xgb_model.json`

## Critical Patterns & Conventions

### Data Pipeline
1. **Raw data** (`data/raw/mmc1.csv`): Original Oberhollenzer dataset
2. **Preprocessing** via `preprocess_funcs.py`: Outlier removal, feature engineering
3. **Model-ready data** (`data/model_ready/`): Train/test splits with engineered features

### Feature Engineering
Standard CPT features are hardcoded in multiple places:
- `utility.py`: `SELECTED_FEATURES` list
- `main.yaml`: `experiment.features` configuration
- Always include: Depth, qc (cone resistance), fs (sleeve friction), Rf, stress parameters, Qtn, Fr

### Model Integration
- **XGBoost**: Native XGBoost library (preferred) with GPU support
- **Scikit-learn models**: KNN, Random Forest, Logistic Regression via sklearn
- **Imbalanced data handling**: SMOTE oversampling + RandomUnderSampler in pipelines
- All models wrapped in `train_predict()` function in `train_eval_funcs.py`

### Configuration Override Pattern
Use Hydra's hierarchical config system:
```python
@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pcfg = Config(**OmegaConf.to_object(cfg))  # Pydantic validation
```

### Format and lint (keep line length = 88, see `pyproject.toml`):

- Format: `uv run ruff format .`
- Sort imports: `uv run isort .`
- Lint (auto-fix where possible): `uv run ruff check --fix .`
- Type check: `uv run mypy src/`

## Dependencies & Environment

### Package Management
- **uv**: Primary package manager (not pip/conda)
- Install: `uv sync` (creates .venv automatically)
- Key dependencies: xgboost, scikit-learn, hydra-core, mlflow, streamlit, optuna

### GPU Support
XGBoost configured for GPU by default (`device: gpu` in xgboost_native.yaml)

## Testing & Validation
- **Rich console output**: Use `Console()` for pretty printing
- **Data validation**: Pydantic models ensure config correctness
- **Model evaluation**: Classification metrics via `evaluate_model()` function
- **Visualization**: Confusion matrices and prediction plots in `plotting.py`

## Common Gotchas
- Soil classification mapping (1-7) defined in `main.yaml` under `experiment.soil_classification`
- Feature names include special characters (e.g., "σ',v (kPa)") - handle carefully in pandas
- Hydra changes working directory - use `Path()` for robust file paths
- MLflow experiments directory must exist before running scripts
