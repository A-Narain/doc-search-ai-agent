from vector_store import collection, embedding_model


def retrieve_chunks(query, k=5, filename_filter=None):

    query_embedding = embedding_model.encode(query)

    # Build optional where filter for ChromaDB
    where_filter = {"filename": filename_filter} if filename_filter else None

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
        where=where_filter   # None = search all docs, string = filter to one file
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved_data = []

    for doc, meta, dist in zip(documents, metadatas, distances):
        retrieved_data.append({
            "text":     doc,
            "filename": meta["filename"],
            "chunk_id": meta["chunk_id"],
            "distance": dist,
            "score":    round(max(0.0, 1 - dist), 4)
        })

    return retrieved_data