import pandas as pd
import numpy as np
import google.generativeai as genai


# Functions for RAG
def find_relevant_content(query, df, top_k=3):
    """Find the most relevant chunks from the DataFrame based on embedding similarity"""
    # Get embedding for the query
    query_embedding_result = genai.embed_content(
        model='models/embedding-001',
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = query_embedding_result['embedding']
    
    # Convert embeddings to numpy arrays for faster computation
    query_embedding_np = np.array(query_embedding)
    
    # Calculate similarity scores (cosine similarity)
    similarities = []
    for _, row in df.iterrows():
        doc_embedding_np = np.array(row['Embedding'])
        
        # Calculate cosine similarity
        similarity = np.dot(doc_embedding_np, query_embedding_np) / (
            np.linalg.norm(doc_embedding_np) * np.linalg.norm(query_embedding_np)
        )
        similarities.append(similarity)
    
    # Add similarities to the dataframe
    df_with_scores = df.copy()
    df_with_scores['similarity'] = similarities
    
    # Sort by similarity and get top k results
    top_results = df_with_scores.sort_values('similarity', ascending=False).head(top_k)
    
    return top_results

def create_dataframe(content_and_embeddings):
    pd.DataFrame([{
        "Text": item["text"],
        "Embedding": item["embedding"]
    } for item in content_and_embeddings])