import ollama as olm
import numpy as np
from pypdf import PdfReader
from pathlib import Path


def chunker(text, chunk_size=180, stepback=30):
    words = text.split()
    chunk_list = []
    step = chunk_size - stepback

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if len(chunk_words) < 30:
            continue

        chunk = " ".join(chunk_words)
        chunk_list.append(chunk)

    return chunk_list


def reader(pdf_path):
    record = PdfReader(pdf_path)
    records = []

    for page_number, page in enumerate(record.pages, start=1):
        text = page.extract_text() or ""
        page_chunks = chunker(text)

        for chunks in page_chunks:
            records.append(
                {
                    "text": chunks,
                    "page number": page_number,
                    "source": Path(pdf_path).name
                }
            )

    return records

def create_embeddings(records, batch_size=32):

    vectors = []

    total = len(records)

    print("Total chunks:", total)

    for start in range(0, total, batch_size):

        batch = records[start:start + batch_size]

        texts = []

        for record in batch:
            texts.append(record["text"])

        response = olm.embed(
            model="embeddinggemma",
            input=texts,
            options={"num_gpu": 0}

        )

        vectors.extend(response["embeddings"])

        completed = min(start + batch_size, total)

        print(
            f"Embedded {completed}/{total} chunks"
        )

    print("Embedding complete.")

    return np.array(vectors)

def load_or_create_embeddings(records, cache_path):

    if Path(cache_path).exists():

        rec_arr = np.load(cache_path)

        if len(rec_arr) == len(records):
            print("Loaded saved embeddings.")
            return rec_arr

        print("Document changed. Recreating embeddings...")


    rec_arr = create_embeddings(records)

    np.save(cache_path, rec_arr)

    print("Embeddings saved.")

    return rec_arr

def embv(question, records, rec_arr, top_k):

    qstn = olm.embed(
        model="embeddinggemma",
        input=question,
        options={"num_gpu": 0}
    )

    qstn_arr = np.array(
        qstn["embeddings"][0]
    )

    scores = rec_arr @ qstn_arr

    big_3 = np.argsort(scores)[::-1][:top_k]

    context = []

    for i in big_3:

        dicti = {
            "text": records[i]["text"],
            "page": records[i]["page number"],
            "source": records[i]["source"],
            "score": float(scores[i])
        }

        context.append(dicti)

    return context
