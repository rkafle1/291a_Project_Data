#!/bin/bash

python pipeline.py --mode build --config configs/config_faiss.yaml

python pipeline.py --mode eval --config configs/config_faiss.yaml --output results/ours_faiss_results.json
