from . import config
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

print("[TF-IDF] Initialisation et vectorisation du corpus...")

# Initialisation du vectorizer avec tes paramètres optimisés
vectorizer = TfidfVectorizer()

# On transforme tout le contenu du corpus chargé dans config.py
doc_matrix = vectorizer.fit_transform(config.CORPUS_DF["content"].tolist())

def retrieve(query: str) -> pd.DataFrame:
    """
    Recherche les chunks les plus pertinents via l'approche TF-IDF (Baseline).
    """
    # 1. Vectorisation de la requête
    query_vec = vectorizer.transform([query])
    
    # 2. Calcul de la similarité cosinus (linear_kernel suffit car les vecteurs sont normés L2)
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
    # 1. Vectorisation groupée de toutes les requêtes d'un coup
    # vectorizer.transform gère nativement les listes de chaînes de caractères
    query_vecs = vectorizer.transform(queries)
    
    # 2. Calcul de la similarité cosinus globale (Produit matriciel)
    # linear_kernel renvoie ici une matrice de dimension (Nb_queries, Nb_documents)
    scores_matrix = linear_kernel(query_vecs, doc_matrix)
    
    results = []
    
    # 3. Extraction des K meilleurs résultats pour chaque requête
    for i in range(len(queries)):
        # On extrait la ligne de scores correspondant à la requête 'i'
        scores = scores_matrix[i]
        
        # argsort trie par ordre croissant, on inverse avec [::-1] puis on coupe à TOP_K
        top_indices = scores.argsort()[::-1][:config.TOP_K]
        
        # Construction du DataFrame de résultat
        result = config.CORPUS_DF.iloc[top_indices].copy().reset_index(drop=True)
        result["score"] = scores[top_indices]
        results.append(result)
        
    return results