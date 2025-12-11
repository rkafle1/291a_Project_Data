#!/bin/bash

python qdrant_retrieval.py

python evaluator2.py --sim-threshold 0.4