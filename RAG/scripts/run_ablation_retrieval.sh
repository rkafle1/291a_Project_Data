#!/bin/bash

python pipeline.py --mode build --config configs/config_qdrant_1.0_0.0.yaml

python pipeline.py --mode eval --config configs/config_qdrant_1.0_0.0.yaml --output results/ablation_retrieval_1.0_0.0_results.json \
    --sim-threshold 0.4

python pipeline.py --mode build --config configs/config_qdrant_0.4_0.6.yaml

python pipeline.py --mode eval --config configs/config_qdrant_0.4_0.6.yaml --output results/ablation_retrieval_0.4_0.6_results.json \
    --sim-threshold 0.4

python pipeline.py --mode build --config configs/config_qdrant_0.55_0.45.yaml

python pipeline.py --mode eval --config configs/config_qdrant_0.55_0.45.yaml --output results/ablation_retrieval_0.55_0.45_results.json \
    --sim-threshold 0.4

python pipeline.py --mode build --config configs/config_qdrant_0.85_0.15.yaml

python pipeline.py --mode eval --config configs/config_qdrant_0.85_0.15.yaml --output results/ablation_retrieval_0.85_0.15_results.json \
    --sim-threshold 0.4
