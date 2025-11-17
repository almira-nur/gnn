@echo off

echo Running uv sync...
uv sync

echo Installing Jupyter kernel...
uv run python -m ipykernel install --user --name gnn --display-name "GNN MLIP"

echo Kernel "gnn" installed successfully.