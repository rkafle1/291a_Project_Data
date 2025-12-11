# CSE 291A: Systems for LLMs and AI Agents Final Project Report

*PyLiRAG: A Hybrid RAG System for PyTorch Lightning Library*

Lai Wei (A69034175), Kaiying Han (A69042727), Richa Kafle (A16964299)

## Installation

We recommend to use conda environment. All of our codes are executed on a Nvidia GPU enabled Ubuntu 22.04 workstation.

```bash
conda create -n rag python=3.10
conda activate rag

pip install -r requirements.txt
```

This would automatically install `faiss-cpu`. To enable GPU support, run:

```bash
pip uninstall faiss-cpu
pip install faiss-gpu
pip install 'numpy<2'  # faiss-gpu is not compatible with numpy>2
```

For qdrant vector database, its recommended to use docker. Run the following command, and `sudo` may be required.

```bash
# For qdrant, we shall run the backend in a docker
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/.cache:/qdrant/storage:z" \
    qdrant/qdrant
```

## Quick Start

### Ours

Get into `RAG` directory. All ours method are executed under this directory.

```bash
cd RAG
```

You can choose to run qdrant or faiss backend. To use the LLM integration in the whole pipeline, first you need to set the `RAG/config/config_<xxx>.yaml` (choose the config file belong to that backend). As default, it requires an API key from [google ai studio](https://aistudio.google.com/app/api-keys), which is free to get. Set it to the environment variable `api_key_env_var`. Don't hardcode it in the code!
```yaml
llm:
  provider: "google"
  model_name: "gemini-2.5-flash" # Options: gemini-1.5-flash, gemini-1.5-pro
  # api_key_env_var: "GEMINI_API_KEY"
  api_key_env_var: "Your_api_Key"
  temperature: 0.1
  max_tokens: 1024
```

If you use qdrant, make sure you have set up the qdrant docker and it's running on [localhost:6333](http://localhost:6333/dashboard#/collections). 

---

To run our method (build database + evaluation), you can run the script:

```bash
# for faiss backend
bash scripts/run_ours_faiss.sh

# for qdrant backend
bash scripts/run_ours_qdrant.sh
```

---

To run our pipeline (e.g., qdrant backend) in interactive query mode and test our RAG system, run:

```bash
# build the database if necessary
python pipeline.py --mode build --config configs/config_qdrant.yaml

# start with interactive query mode
python pipeline.py --mode query
```

The later command will create a interactive interface for you to input the query, it will retrieve top-5 result and generate a answer using LLM for the query. A sampel output should be like in the [sample_output](./RAG/utils/sample_output.txt)

You may also see the collections of qdrant after finishing embedding on the [qdrant portal](http://localhost:6333/dashboard#/collections).

![collections](./figs/Collections.png)

---

To run ablation experiment, run:

```bash
bash scripts/run_ablation_retrieval.sh
```

### baselines

You may choose to run qdrant or faiss baseline. Get into the corresponding directory:

```bash
# faiss
cd baselines/faiss
bash run_baseline_faiss.sh
```

```bash
# qdrant
cd baselines/qdrant
bash run_baseline_qdrant.sh
```

## Project Structure

The followings shows ours pipeline method:

```
RAG/
├── configs/
│   └── config.yaml          # Main configuration file
├── chunking/
│   ├── ast_chunker.py       # AST-based code chunking
│   └── recursive_chunker.py # Recursive text chunking
├── embeddings/
│   └── code_embedder.py     # UniXcoder embeddings
├── storage/
│   ├── vector_store.py      # FAISS/Qdrant vector store
│   └── graph_db.py          # Repository Semantic Graph
├── retrieval/
│   └── hybrid_retriever.py  # Hybrid & RepoCoder retrieval
├── evaluation/
│   └── evaluator.py         # Evaluation metrics
├── LLM/
│   └── LLM.py               # LLM generation
├── utils/
│   └── data_utils.py        # Data loading utilities
├── pipeline.py              # Main RAG pipeline
├── run_baseline_comparison.py # Baseline comparison script
├── requirements.txt
└── README.md
```

## References

- [UniXcoder](https://github.com/microsoft/CodeBERT) - Cross-modal code representation
- [RepoCoder](https://arxiv.org/abs/2303.12570) - Repository-level code retrieval
- [RepoHyper](https://arxiv.org/abs/2403.06095) - Hybrid retrieval for code
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search
- [Qdrant](https://qdrant.tech/) - Vector database
