import json
import os
import pandas as pd
from transformers import AutoTokenizer


def _chunk_text(text, tokenizer, max_tokens, overlap):
    tokens = tokenizer.encode(text)
    chunks = []
    i = 0

    while i < len(tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
        i += max_tokens - overlap

    return chunks


def to_chunk(input_path, output_filename, tokenizer_name, max_tokens, overlap):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    with open(input_path, "r", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f]

    chunked = []
    rows = []

    for doc in corpus:
        text = doc.get("text", "")
        chunks = _chunk_text(text, tokenizer, max_tokens, overlap)

        for i, chunk in enumerate(chunks):
            new_doc = {
                "doc_id": f"{doc['doc_id']}_chunk_{i}",
                "source": doc.get("source"),
                "title": doc.get("title"),
                "url": doc.get("url"),
                "text": chunk,
                "metadata": {
                    **doc.get("metadata", {}),
                    "chunk_index": i,
                    "num_chunks": len(chunks)
                }
            }

            chunked.append(new_doc)

            # Format compatible avec load_corpus()
            rows.append({
                "doc_id": new_doc["doc_id"],
                "url": new_doc["url"],
                "content": new_doc["title"] + "\n" + new_doc["text"]
            })

    output_path = os.path.join("data", output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in chunked:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    return pd.DataFrame(rows)