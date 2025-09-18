import os
import chromadb
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import time

# --- 1. SETUP ---
DOCUMENTS_FOLDER = "./my_documents"

# --- 2. LOAD AND CHUNK DOCUMENTS ---
print("Scanning document folder...")
all_chunks = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

for filename in os.listdir(DOCUMENTS_FOLDER):
    file_path = os.path.join(DOCUMENTS_FOLDER, filename)
    if not os.path.isfile(file_path):
        continue
    try:
        loader = None
        if filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.lower().endswith('.pptx'):
            loader = UnstructuredPowerPointLoader(file_path)
        elif filename.lower().endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        
        if loader:
            print(f" > Processing {filename}...")
            documents = loader.load()
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
            print(f"   ...found {len(chunks)} chunks.")
        else:
            print(f" > Skipping unsupported file: {filename}")
    except Exception as e:
        print(f"   Error processing {filename}: {e}")

print(f"\n✅ Total chunks created from all documents: {len(all_chunks)}")

# --- 3. GENERATE EMBEDDINGS ---
if all_chunks:
    print("\nGenerating embeddings... (This may take a moment)")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    documents_text = [chunk.page_content for chunk in all_chunks]
    
    start_time = time.time()
    embeddings = embedding_model.encode(documents_text, show_progress_bar=True)
    end_time = time.time()
    
    print(f"✅ Embeddings generated in {end_time - start_time:.2f} seconds.")

    # --- 4. STORE IN VECTOR DATABASE (ChromaDB) ---
    print("\nSetting up ChromaDB vector store...")
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        if "my_study_collection" in [c.name for c in client.list_collections()]:
            print("   Clearing old database collection...")
            client.delete_collection(name="my_study_collection")
    except Exception as e:
        print(f"   Error clearing collection: {e}")
        
    collection = client.get_or_create_collection(name="my_study_collection")
    
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    metadatas = [
        {"source": chunk.metadata.get("source", "").split('/')[-1], "page": chunk.metadata.get("page", 0)} 
        for chunk in all_chunks
    ]
    
    print("Adding embeddings to the collection in batches...")
    
    # ###############################################################
    # ## BATCHING LOGIC START (This is the new part) ##
    # ###############################################################
    batch_size = 1000
    for i in range(0, len(all_chunks), batch_size):
        # Find the end of the batch
        end_index = min(i + batch_size, len(all_chunks))
        
        # Extract the batch data
        batch_embeddings = embeddings[i:end_index].tolist()
        batch_documents = documents_text[i:end_index]
        batch_metadatas = metadatas[i:end_index]
        batch_ids = ids[i:end_index]
        
        # Add the batch to the collection
        collection.add(
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        print(f"   ...added batch {i//batch_size + 1} ({end_index}/{len(all_chunks)})")
    # ###############################################################
    # ## BATCHING LOGIC END ##
    # ###############################################################

    print("\n🎉 Success! Your documents have been embedded and stored in ChromaDB.")
    print(f"Total documents in collection: {collection.count()}")
else:
    print("\nNo documents were processed. Please add supported files to the 'my_documents' folder.")