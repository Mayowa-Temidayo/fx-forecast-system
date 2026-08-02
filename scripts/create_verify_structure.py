"""
Create the verification script structure.

Run:
    uv run python scripts/create_verify_structure.py
"""

from pathlib import Path

VERIFY_DIR = Path("scripts") / "verify"

FILES = [
    "__init__.py",
    "verify_io.py",
    "verify_fetch.py",
    "verify_validate.py",
    "verify_preprocess.py",
    "verify_features.py",
    "verify_models.py",
    "verify_backtest.py",
]


def create_file(path: Path) -> None:
    """Create an empty file if it doesn't already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.write_text("", encoding="utf-8")
        print(f"✓ Created {path}")
    else:
        print(f"• Exists   {path}")


def main() -> None:
    print("Creating verification structure...\n")

    for filename in FILES:
        create_file(VERIFY_DIR / filename)

    print("\nVerification structure ready.")


if __name__ == "__main__":
    main()
