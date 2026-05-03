from . import config
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

print("[TF-IDF] Initialisation et vectorisation du corpus...")

# Initialisation du vectoriseur TF-IDF
vectorizer = TfidfVectorizer(
)

doc_matrix = vectorizer.fit_transform(config.CORPUS_DF["content"].tolist())

def retrieve(query: str) -> pd.DataFrame:
    """
    Recherche les chunks les plus pertinents pour plusieurs requêtes avec TF-IDF.
    """
    # 1. Vectorisation de la requête
    query_vec = vectorizer.transform([query])
    
    # 2. Calcul de la similarité cosinus
    scores = linear_kernel(query_vec, doc_matrix).flatten()
    
    # 3. Récupération des TOP_K indices
    top_indices = scores.argsort()[::-1][:config.TOP_K]
    
    # 4. Construction du DataFrame de sortie
    result = config.CORPUS_DF.iloc[top_indices].copy().reset_index(drop=True)
    result["score"] = scores[top_indices]
    
    return result

def retrieve_batch(queries: list[str]) -> list[pd.DataFrame]:
    """
    Recherche les chunks les plus pertinents pour une liste de requêtes en une seule passe via TF-IDF.
    """
    # 1. Vectorisation groupée des requêtes
    query_vecs = vectorizer.transform(queries)
    
    # 2. Calcul global de la similarité cosinus
    scores_matrix = linear_kernel(query_vecs, doc_matrix)
    
    results = []
    
    # 3. Extraction des K meilleurs résultats pour chaque requête
    for i in range(len(queries)):
        # Extraction des scores associés à la requête courante
        scores = scores_matrix[i]
        
        # Tri décroissant des scores et conservation des TOP_K meilleurs résultats
        top_indices = scores.argsort()[::-1][:config.TOP_K]
        
        # Construction du DataFrame de résultat
        result = config.CORPUS_DF.iloc[top_indices].copy().reset_index(drop=True)
        result["score"] = scores[top_indices]
        results.append(result)
        
    return results