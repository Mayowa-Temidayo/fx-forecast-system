Write-Host "Running Ruff..."
uv run ruff check .

Write-Host "Running Pyright..."
uv run pyright

Write-Host "Running Pytest..."
uv run pytest

Write-Host "`n✅ Quality gate completed."