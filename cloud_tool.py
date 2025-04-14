import google.generativeai as genai
from const import AI_MODEL, EMBEDDING_MODEL


def create_model_config():
    # Create a new model instance
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }

    response_model = genai.GenerativeModel(
        model_name=AI_MODEL,
        generation_config=generation_config,
    )

    return response_model

def create_embedding(text):
    try:
        embedding = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_query"
        )
        return embedding
    except Exception as e:
        raise ValueError(f"Error creating embedding: {str(e)}")