from . import config
import pandas as pd
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import spacy
nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])

def spacy_tokenizer(text: str) -> list[str]:
    doc = nlp(str(text).lower())

    tokens = [
        token.text
        for token in doc
    ]

    bigrams = [
        tokens[i] + "_" + tokens[i + 1]
        for i in range(len(tokens) - 1)
    ]

    return tokens + bigrams

print("[BM25] Indexation du corpus...")

docs = [
    Document(
        page_content=row["content"],
        metadata={"row_id": int(i)}
    )
    for i, row in config.CORPUS_DF.iterrows()
]

bm25_retriever = BM25Retriever.from_documents(
    docs,
    preprocess_func=spacy_tokenizer,
)

bm25_retriever.k = config.TOP_K


def retrieve(query: str) -> pd.DataFrame:
    results = bm25_retriever.invoke(query)
    row_ids = [doc.metadata["row_id"] for doc in results]
    return config.CORPUS_DF.iloc[row_ids].copy().reset_index(drop=True)


def retrieve_batch(queries: list[str]) -> list[pd.DataFrame]:
    batch_results = bm25_retriever.batch(queries)
    final_results = []

    for results in batch_results:
        row_ids = [doc.metadata["row_id"] for doc in results]
        final_results.append(
            config.CORPUS_DF.iloc[row_ids].copy().reset_index(drop=True)
        )

    return final_results