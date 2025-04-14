import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
from datetime import datetime


# Set page config
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

# Function to clear chat history
def clear_chat():
    st.session_state.messages = []

# App title
st.title("PDF Q&A Chatbot")

# Sidebar for API key and PDF upload
with st.sidebar:
    st.header("Settings")
    # Try to get API key from secrets, handle case when not available
    try:
        secret = st.secrets["GEMINI_API_KEY"]
        has_secret = True
    except Exception:
        secret = ""
        has_secret = False
    
    # If we have a stored key in session state, use that, 
    # otherwise use the secret if available
    if 'api_key' in st.session_state:
        api_key = st.session_state.api_key
    elif has_secret:
        api_key = secret
        st.session_state.api_key = api_key
    else:
        api_key = ""
        st.warning("No API key found in Streamlit secrets. Please enter your key below.")
    
    # api_key = st.text_input("Enter your Google API Key:", 
    #                        type="password",
    #                        value=default_value)
    
    if api_key:
        st.session_state.api_key = api_key
        # Configure the Google API
        genai.configure(api_key=api_key)
    else:
        st.warning("Please provide a Google API key to continue.")
    
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
        
        # Process button
        if st.button("Process PDF"):
            with st.spinner("Processing PDF... This may take a while."):
                try:
                    # Load the PDF
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()
                    
                    # Split text into chunks
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    texts = text_splitter.split_documents(documents)
                    text_contents = [doc.page_content for doc in texts]
                    
                    st.info(f"Split PDF into {len(text_contents)} chunks")
                    
                    # Create embeddings
                    model = 'models/embedding-001'
                    content_and_embeddings = []
                    
                    for i, text in enumerate(text_contents):
                        embedding_result = genai.embed_content(
                            model=model,
                            content=text,
                            task_type="retrieval_query"
                        )
                        embedding_values = embedding_result['embedding']
                        content_and_embeddings.append({
                            "text": text,
                            "embedding": embedding_values
                        })
                    
                    # Create DataFrame
                    df = pd.DataFrame([{
                        "Text": item["text"],
                        "Embedding": item["embedding"]
                    } for item in content_and_embeddings])
                    
                    # Store in session state
                    st.session_state.df = df
                    st.session_state.pdf_processed = True
                    st.success(f"PDF processed successfully! Created {len(df)} embeddings.")
                
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                finally:
                    # Clean up the temporary file
                    os.unlink(pdf_path)
    
    # Display information about the PDF status
    if st.session_state.pdf_processed:
        st.success("PDF is processed and ready for questions!")
    else:
        st.info("Please upload and process a PDF before asking questions.")

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

def answer_question_with_rag(query, df):
    """Answer a question using RAG (Retrieval Augmented Generation)"""
    # Get most relevant content chunks
    relevant_chunks = find_relevant_content(query, df, top_k=3)
    
    # Extract the text from the relevant chunks
    context_texts = relevant_chunks['Text'].tolist()
    
    # Join the texts with separators
    context = "\n\n---\n\n".join(context_texts)
    
    # Create prompt with context
    prompt = f"""
    You are a helpful assistant that answers questions based on the provided context.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {query}
    
    ANSWER:
    """
    
    # Create a new model instance
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
    
    response_model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
    )
    
    # Generate response
    response = response_model.generate_content(prompt)
    
    return {
        'answer': response.text,
        'relevant_chunks': relevant_chunks[['Text', 'similarity']].to_dict('records')
    }

# Chat interface
# st.header("Ask questions about your PDF")

# Add Clear Chat button
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("Clear Chat"):
        clear_chat()
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input field for new question
if query := st.chat_input("Ask a question about the uploaded PDF"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.write(query)
    
    # Check if PDF has been processed
    if not st.session_state.pdf_processed:
        with st.chat_message("assistant"):
            st.write("Please upload and process a PDF first before asking questions.")
        st.session_state.messages.append({"role": "assistant", "content": "Please upload and process a PDF first before asking questions."})
    else:
        # Process the query and generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = answer_question_with_rag(query, st.session_state.df)
                st.write(result['answer'])
                
                # Show relevant chunks in an expander (optional)
                with st.expander("View source passages"):
                    for i, chunk in enumerate(result['relevant_chunks']):
                        st.markdown(f"**Passage {i+1}** (Similarity: {chunk['similarity']:.4f})")
                        st.markdown(chunk['Text'][:500] + "...")
                        st.divider()
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": result['answer']})

# Footer with copyright notice
# Persistent footer at the bottom of the screen
current_year = datetime.now().year
st.markdown(
    f"""
    <style>
    .footer {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #fafafa;
        text-align: center;
        font-size: 0.8em;
        color: #666;
        z-index: 100;
        border-top: 1px solid #ddd;
    }}
    </style>
    <div class="footer">
        &copy; {current_year} PDF Q&A Chatbot. All rights reserved.<br>
        <small>Developer: <a href="javascript:void(0);" 
        title="Abhisek Soni" style="text-decoration:none;color:#888;">AS</a></small>
    </div>
    """,
    unsafe_allow_html=True
)