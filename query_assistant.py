import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama

# --- Page Configuration ---
st.set_page_config(page_title="My Study Assistant", page_icon="📚", layout="wide")

# --- Functions for Caching ---
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

@st.cache_data
def get_available_classes(_collection):
    """Gets all unique class names from the collection's metadata."""
    print("Fetching available classes...")
    all_metadata = _collection.get(include=["metadatas"])
    unique_classes = set(item['class'] for item in all_metadata['metadatas'])
    return ["All Classes"] + sorted(list(unique_classes))

@st.cache_resource
def load_llm():
    """Loads the Ollama LLM."""
    print("Loading LLM...")
    return Ollama(model="llama3:8b")

# --- Main App ---
st.title("My AI Study Assistant 📚")
st.write("Ask a question about your documents, and the AI will synthesize an answer.")

# --- Load Models and DB ---
model = load_embedding_model()
collection = load_vector_db()
llm = load_llm()
available_classes = get_available_classes(collection)

# --- UI Elements ---
col1, col2 = st.columns([3, 1])
with col1:
    question = st.text_input("Enter your question here:", placeholder="e.g., What are the main types of real-time systems?")
with col2:
    selected_class = st.selectbox("Filter by class:", options=available_classes)

# --- Main Logic ---
if question:
    with st.spinner("Searching for relevant documents..."):
        # Build the where filter
        where_filter = {}
        if selected_class != "All Classes":
            where_filter = {"class": selected_class}
        
        # Get embeddings and query the database
        question_embedding = model.encode(question).tolist()
        results = collection.query(query_embeddings=[question_embedding], n_results=10, where=where_filter)
        
        # Prepare context for the LLM
        context = "\n\n---\n\n".join(results['documents'][0])

    if not context:
        st.warning("Could not find any relevant documents in the selected class to answer your question.")
    else:
        with st.spinner("Synthesizing answer..."):
            # Create the prompt for the LLM
            prompt_template = f"""
            You are an expert study assistant. Your goal is to provide a clear and concise answer to the user's question based ONLY on the context provided.
            Do not use any outside knowledge. If the context does not contain the answer, state that you cannot answer the question with the given information.

            CONTEXT:
            {context}

            QUESTION:
            {question}

            ANSWER:
            """
            
            # Get the answer from the LLM
            answer = llm.invoke(prompt_template)
            
            # Display the answer
            st.markdown("### Answer")
            st.success(answer)
            
            # Display the sources used
            with st.expander("Show Sources"):
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    source = metadata.get('source', 'Unknown Source')
                    page = metadata.get('page', 'N/A')
                    st.info(f"**Source {i+1} (from __{source}__, Page {page}):**\n\n" + doc)