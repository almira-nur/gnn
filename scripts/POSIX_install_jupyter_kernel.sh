#!/bin/bash
set -e

uv sync
uv run python -m ipykernel install --user --name gnn --display-name "GNN MLIP"