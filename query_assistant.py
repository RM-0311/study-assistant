import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

# --- Page Configuration ---
st.set_page_config(
    page_title="My Study Assistant",
    page_icon="📚",
    layout="wide"
)

# --- Functions for Caching ---
# Streamlit will cache the results of these functions to avoid reloading the model
# and database on every interaction, making the app much faster.

@st.cache_resource
def load_embedding_model():
    """Loads the SentenceTransformer model."""
    print("Loading embedding model...")
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_vector_db():
    """Loads the ChromaDB collection."""
    print("Loading vector database...")
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection(name="my_study_collection")

# --- Main App ---
st.title("My AI Study Assistant 📚")
st.write("Ask a question about your document, and the AI will find the most relevant paragraphs.")

# --- Load Model and DB ---
model = load_embedding_model()
collection = load_vector_db()

# --- User Input ---
question = st.text_input(
    "Enter your question here:",
    placeholder="e.g., What are the main types of real-time systems?"
)

if question:
    st.write("### Results:")
    
    # 1. Convert the user's question into an embedding
    question_embedding = model.encode(question).tolist()
    
    # 2. Query the vector database
    # Ask for the 5 most similar chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=5
    )
    
    # 3. Display the results
    if not results['documents'][0]:
        st.warning("No relevant documents found.")
    else:
        for i, doc in enumerate(results['documents'][0]):
            page_number = results['metadatas'][0][i]['page']
            
            st.markdown(f"**Result {i+1} (from Page {page_number}):**")
            st.info(doc) # Display the document chunk in a blue box
            st.divider() # Add a separator line