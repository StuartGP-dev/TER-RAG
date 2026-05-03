import os
import json
import torch
import pandas as pd
from . import chunker

# -----------------------------
# VARIABLES GLOBALES ET PROXY
# -----------------------------
USE_UNIV_PROXY = False

if USE_UNIV_PROXY:
    PROXY = "http://cache.ha.univ-nantes.fr:3128/"
    os.environ["http_proxy"] = PROXY
    os.environ["https_proxy"] = PROXY
    os.environ["HTTP_PROXY"] = PROXY
    os.environ["HTTPS_PROXY"] = PROXY


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
HF_TOKEN = os.getenv("HF_TOKEN")

# -----------------------------
# CHEMINS DES FICHIERS ET MODÈLES
# -----------------------------
JSONL_PATH = "data/documents_all.jsonl"
GENERATOR_MODEL_PATH = "/home/partage/Mistral-7B-Instruct-v0.3"
EMBEDDING_MODEL_PATH = "BAAI/bge-m3"
DPR_CONTEXT_MODEL_PATH = "etalab-ia/dpr-ctx_encoder-fr_qa-camembert"
DPR_QUESTION_MODEL_PATH = "etalab-ia/dpr-question_encoder-fr_qa-camembert"

# -----------------------------
# PARAMÈTRES RAG
# -----------------------------
TOP_K = 5
DPR_BATCH_SIZE = 16
DEVICE = "cuda:0"

CHUNKED_PATH = "data/documents_all_chunked.jsonl"

if os.path.exists(CHUNKED_PATH) and os.path.getsize(CHUNKED_PATH) > 0:
    print("[CONFIG] Fichier chunké déjà présent, pas de rechunking.")
    CORPUS_DF = pd.read_json(CHUNKED_PATH, lines=True)

    CORPUS_DF = pd.DataFrame({
        "doc_id": CORPUS_DF["doc_id"],
        "url": CORPUS_DF["url"],
        "content": CORPUS_DF["title"].fillna("") + "\n" + CORPUS_DF["text"].fillna("")
    })

else:
    print("[CONFIG] Fichier chunké absent ou vide, chunking...")
    CORPUS_DF = chunker.to_chunk(
        JSONL_PATH,
        "documents_all_chunked.jsonl",
        GENERATOR_MODEL_PATH,
        max_tokens=450,
        overlap=80
    )