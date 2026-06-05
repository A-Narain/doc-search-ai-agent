from vector_store import (
    collection,
    embedding_model
)


def retrieve_chunks(query, k=5, filename_filter=None):

    query_embedding = embedding_model.encode(query)

    where = {"filename": filename_filter} if filename_filter else None

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved_data = []

    for doc, meta, dist in zip(documents, metadatas, distances):

        # Convert L2 distance → bounded similarity score (0.0 – 1.0)
        # dist=0 → score=1.0 (perfect match), dist→∞ → score→0.0
        score = 1 / (1 + dist)

        retrieved_data.append(
            {
                "text":     doc,
                "filename": meta["filename"],
                "chunk_id": meta["chunk_id"],
                "score":    round(score, 4)
            }
        )

    return retrieved_data