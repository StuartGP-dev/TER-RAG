import os
from . import config
os.environ["TRANSFORMERS_VERBOSITY"] = "info"
import numpy as np
import pandas as pd
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
# On masque les alertes de TQDM pour éviter le rouge
warnings.filterwarnings("ignore", message=".*IProgress not found.*")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if config.HF_TOKEN is not None:
    os.environ["HF_TOKEN"] = config.HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = config.HF_TOKEN
import torch # Ajout nécessaire pour le dtype de calcul
from transformers import BitsAndBytesConfig

# 1. Configuration de la quantification 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # Type NF4 : meilleur pour la précision
    bnb_4bit_compute_dtype=torch.bfloat16 # Accélère le calcul (utilise float16 si ta carte est ancienne)
)

print("[HyDE] Chargement du LLM quantifié...")

generator = pipeline(
    "text-generation",
    model=config.GENERATOR_MODEL_PATH,
    tokenizer=config.GENERATOR_MODEL_PATH,
    # Remplacer device=0 par device_map="auto" pour laisser accelerate gérer la VRAM
    device_map="auto", 
    model_kwargs={"quantization_config": bnb_config},
    token=config.HF_TOKEN,
    max_length=None,
    do_sample=False
)


# Force le tokenizer à utiliser le jeton de fin (EOS) pour le padding
generator.tokenizer.pad_token_id = generator.tokenizer.eos_token_id

embedder = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL_PATH,
    model_kwargs={"device": config.DEVICE},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 128},
)

print("[HyDE] Pré-calcul des embeddings du corpus...")
doc_embeddings = np.array(
    embedder.embed_documents(config.CORPUS_DF["content"].astype(str).tolist()),
    dtype=np.float32
)
#doc_embeddings /= np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-12

def retrieve(query: str) -> pd.DataFrame:
    """
    Génère un faux document avec Llama 3, puis recherche sémantiquement les vrais chunks.
    """
    # 1. Génération du document hypothétique
    messages = [
        {"role": "system", "content": "Tu es un expert. Rédige un court passage scientifique factuel qui répondrait directement à la question posée. Ne fais pas d'introduction."},
        {"role": "user", "content": f"Question: {query}\nPassage:"}
    ]
    prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = generator(prompt, max_new_tokens=100 , pad_token_id=generator.tokenizer.eos_token_id, eos_token_id=generator.tokenizer.eos_token_id,return_full_text=False)
    hypothetical_doc = outputs[0]["generated_text"].strip()
    
    # 2. Encodage et calcul de similarité Cosinus
    query_embedding = np.array(embedder.embed_query(hypothetical_doc), dtype=np.float32)
    query_embedding /= np.linalg.norm(query_embedding) + 1e-12
    
    scores = doc_embeddings @ query_embedding
    idx = np.argpartition(scores, -config.TOP_K)[-config.TOP_K:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    
    result = config.CORPUS_DF.iloc[idx].copy().reset_index(drop=True)
    result["score"] = scores[idx]
    
    return result

def retrieve_batch(queries: list[str]) -> list[pd.DataFrame]:
    """
    Version optimisée pour traiter plusieurs requêtes d'un coup.
    """
    batch_size = 4
    # 1. Préparation de tous les prompts
    all_prompts = []
    for query in queries:
        messages = [
            {"role": "system", "content": "Tu es un expert. Rédige un court passage scientifique factuel qui répondrait directement à la question posée. Ne fais pas d'introduction."},
            {"role": "user", "content": f"Question: {query}\nPassage:"}
        ]
        prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompts.append(prompt)

    # 2. Génération groupée (Batch Generation)
    # On ajoute max_length=None pour supprimer définitivement le warning
    outputs = generator(
        all_prompts, 
        max_new_tokens=100, 
        batch_size=batch_size, 
        pad_token_id=generator.tokenizer.eos_token_id,
        return_full_text=False,
        do_sample=False,
        num_beams=1,
        max_length=None
    )
    
    hypothetical_docs = [out[0]["generated_text"].strip() for out in outputs]

    # 3. Encodage groupé des documents hypothétiques
    # HuggingFaceEmbeddings gère déjà le batching en interne
    query_embeddings = np.array(embedder.embed_documents(hypothetical_docs), dtype=np.float32)

    # 4. Calcul de similarité pour chaque requête (Produit matriciel)
    # doc_embeddings: (N_docs, Dim) | query_embeddings.T: (Dim, N_queries)
    all_scores = doc_embeddings @ query_embeddings.T  # Résultat: (N_docs, N_queries)

    results_list = []
    for i in range(len(queries)):
        scores = all_scores[:, i]
        idx = np.argpartition(scores, -config.TOP_K)[-config.TOP_K:]
        idx = idx[np.argsort(scores[idx])[::-1]]        
        result = config.CORPUS_DF.iloc[idx].copy().reset_index(drop=True)
        result["score"] = scores[idx]
        results_list.append(result)

    return results_list