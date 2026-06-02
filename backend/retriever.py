from vector_store import (
    collection,
    embedding_model
)


def retrieve_chunks(query, k=5):

    query_embedding = embedding_model.encode(
        query
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    retrieved_data = []

    for doc, meta in zip(
        documents,
        metadatas
    ):

        retrieved_data.append(
            {
                "text": doc,
                "filename": meta["filename"],
                "chunk_id": meta["chunk_id"]
            }
        )

    return retrieved_data