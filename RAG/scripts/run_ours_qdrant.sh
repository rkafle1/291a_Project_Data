#!/bin/bash

python pipeline.py --mode build --config configs/config_qdrant.yaml

python pipeline.py --mode eval --config configs/config_qdrant.yaml --output results/ours_qdrant_results.json
