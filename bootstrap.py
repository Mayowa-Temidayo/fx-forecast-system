"""
Bootstrap the FX Forecast System project structure.

Run:
    uv run bootstrap.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------

DIRECTORIES = [
    ".github/workflows",
    "configs",
    "data/raw",
    "data/interim",
    "data/processed",
    "data/external",
    "data/features",
    "docs",
    "experiments",
    "models",
    "notebooks",
    "outputs/figures",
    "outputs/predictions",
    "outputs/reports",
    "scripts",
    "src/fx_forecast",
    "src/fx_forecast/config",
    "src/fx_forecast/data",
    "src/fx_forecast/features",
    "src/fx_forecast/models",
    "src/fx_forecast/evaluation",
    "src/fx_forecast/inference",
    "src/fx_forecast/deployment",
    "src/fx_forecast/utils",
    "tests",
]


# ---------------------------------------------------------------------
# Python template
# ---------------------------------------------------------------------

PYTHON_TEMPLATE = '''"""
{title}
"""

from __future__ import annotations

# TODO: Implement.
'''


# ---------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------

FILES = {
    ".gitignore": "",
    "README.md": "# FX Forecast System\n",
    "pyproject.toml": "",
    "configs/default.yaml": "",
    "configs/development.yaml": "",
    "configs/production.yaml": "",
    ".github/workflows/ci.yml": "",
    "notebooks/01_data_pipeline.ipynb": "{}",
    "notebooks/02_feature_engineering.ipynb": "{}",
    "notebooks/03_model_development.ipynb": "{}",
    "notebooks/04_evaluation_deployment.ipynb": "{}",
    "scripts/__init__.py": "",
    "tests/__init__.py": "",
}

PYTHON_MODULES = {
    "src/fx_forecast/__init__.py": "FX Forecast package",
    "src/fx_forecast/config/__init__.py": "Configuration package",
    "src/fx_forecast/config/settings.py": "Application settings",
    "src/fx_forecast/config/paths.py": "Project paths",
    "src/fx_forecast/config/constants.py": "Project constants",
    "src/fx_forecast/data/__init__.py": "Data package",
    "src/fx_forecast/data/fetch.py": "Data acquisition",
    "src/fx_forecast/data/clean.py": "Data cleaning",
    "src/fx_forecast/data/validate.py": "Data validation",
    "src/fx_forecast/data/split.py": "Dataset splitting",
    "src/fx_forecast/features/__init__.py": "Feature engineering package",
    "src/fx_forecast/features/technical.py": "Technical indicators",
    "src/fx_forecast/features/statistical.py": "Statistical features",
    "src/fx_forecast/features/calendar.py": "Calendar features",
    "src/fx_forecast/features/target.py": "Target generation",
    "src/fx_forecast/features/selection.py": "Feature selection",
    "src/fx_forecast/models/__init__.py": "Model package",
    "src/fx_forecast/models/base.py": "Base model interface",
    "src/fx_forecast/models/linear.py": "Linear models",
    "src/fx_forecast/models/tree.py": "Tree models",
    "src/fx_forecast/models/boosting.py": "Boosting models",
    "src/fx_forecast/models/classical.py": "Classical forecasting models",
    "src/fx_forecast/models/deep_learning.py": "Deep learning models",
    "src/fx_forecast/models/ensemble.py": "Ensemble models",
    "src/fx_forecast/evaluation/__init__.py": "Evaluation package",
    "src/fx_forecast/evaluation/metrics.py": "Evaluation metrics",
    "src/fx_forecast/evaluation/backtest.py": "Backtesting",
    "src/fx_forecast/evaluation/diagnostics.py": "Model diagnostics",
    "src/fx_forecast/evaluation/explainability.py": "Explainability",
    "src/fx_forecast/inference/__init__.py": "Inference package",
    "src/fx_forecast/inference/predictor.py": "Prediction pipeline",
    "src/fx_forecast/inference/postprocess.py": "Prediction post-processing",
    "src/fx_forecast/deployment/__init__.py": "Deployment package",
    "src/fx_forecast/deployment/app.py": "Streamlit application",
    "src/fx_forecast/utils/__init__.py": "Utilities package",
    "src/fx_forecast/utils/logger.py": "Logging utilities",
    "src/fx_forecast/utils/helpers.py": "Helper functions",
    "src/fx_forecast/utils/io.py": "Input/output helpers",
    "src/fx_forecast/utils/plotting.py": "Plotting utilities",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def create_directories() -> None:
    for directory in DIRECTORIES:
        path = ROOT / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"[DIR ] {directory}")


def create_files() -> None:
    for file_name, content in FILES.items():
        path = ROOT / file_name

        if path.exists():
            print(f"[SKIP] {file_name}")
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"[FILE] {file_name}")


def create_python_modules() -> None:
    for file_name, title in PYTHON_MODULES.items():
        path = ROOT / file_name

        if path.exists():
            print(f"[SKIP] {file_name}")
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            PYTHON_TEMPLATE.format(title=title),
            encoding="utf-8",
        )
        print(f"[PY  ] {file_name}")


def main() -> None:
    print("\nCreating project structure...\n")

    create_directories()
    create_files()
    create_python_modules()

    print("\nDone.")
    print("\nProject successfully bootstrapped.\n")


if __name__ == "__main__":
    main()
