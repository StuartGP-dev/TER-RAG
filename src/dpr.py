from . import config
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging as hf_logging
import os

if config.HF_TOKEN is not None:
    os.environ["HF_TOKEN"] = config.HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = config.HF_TOKEN
print("[DPR] Chargement des encodeurs...")

old_verbosity = hf_logging.get_verbosity()
hf_logging.set_verbosity_error()

ctx_tokenizer = AutoTokenizer.from_pretrained(config.DPR_CONTEXT_MODEL_PATH)
ctx_encoder = AutoModel.from_pretrained(config.DPR_CONTEXT_MODEL_PATH).to(config.DEVICE)

q_tokenizer = AutoTokenizer.from_pretrained(config.DPR_QUESTION_MODEL_PATH)
q_encoder = AutoModel.from_pretrained(config.DPR_QUESTION_MODEL_PATH).to(config.DEVICE)

hf_logging.set_verbosity(old_verbosity)

print("[DPR] Encodage du corpus (batching)...")
dpr_batches = []
texts = config.CORPUS_DF["content"].tolist()

for i in range(0, len(texts), config.DPR_BATCH_SIZE):
    batch_texts = texts[i : i + config.DPR_BATCH_SIZE]
    batch_inputs = ctx_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(config.DEVICE)
    with torch.no_grad():
        # Extraction de l'embedding du premier token
        outputs = ctx_encoder(**batch_inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
    dpr_batches.append(batch_embeddings)
    
doc_embeddings = np.vstack(dpr_batches)
doc_embeddings /= np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-12

def retrieve(query: str) -> pd.DataFrame:
    """
    Encode la question avec l'encodeur spécifique et trouve les chunks correspondants.
    """
    inputs = q_tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(config.DEVICE)
    
    with torch.no_grad():
        # Extraction de l'embedding du premier token pour la requête
        outputs = q_encoder(**inputs)
        query_embedding = outputs.last_hidden_state[0, 0, :].detach().cpu().numpy().astype(np.float32)
        
    query_embedding /= np.linalg.norm(query_embedding) + 1e-12
    scores = doc_embeddings @ query_embedding
    idx = np.argsort(scores)[-config.TOP_K:][::-1]
    
    result = config.CORPUS_DF.iloc[idx].copy().reset_index(drop=True)
    result["score"] = scores[idx]
    
    return result

def retrieve_batch(queries: list[str]) -> list[pd.DataFrame]:
    """
    Encode une liste de questions en une seule passe avec l'encodeur spécifique 
    et trouve les chunks correspondants pour chacune.
    """
    # Encodage groupé des questions avec padding pour gérer les longueurs variables
    inputs = q_tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512).to(config.DEVICE)
    
    with torch.no_grad():
        # Extraction des embeddings du premier token pour tout le batch
        outputs = q_encoder(**inputs)
        query_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
        
    # Normalisation de chaque embedding de requête
    query_embeddings /= np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-12
    
    # Calcul matriciel des similarités entre documents et requêtes
    all_scores = doc_embeddings @ query_embeddings.T
    
    results = []
    # Construction du DataFrame de résultats pour chaque question
    for i in range(len(queries)):
        scores = all_scores[:, i]
        idx = np.argsort(scores)[-config.TOP_K:][::-1]
        
        result = config.CORPUS_DF.iloc[idx].copy().reset_index(drop=True)
        result["score"] = scores[idx]
        results.append(result)
        
    return results