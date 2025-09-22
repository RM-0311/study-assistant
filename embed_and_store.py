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

# Use os.walk to go through all subdirectories
for subdir, dirs, files in os.walk(DOCUMENTS_FOLDER):
    for filename in files:
        file_path = os.path.join(subdir, filename)
        
        # The "class name" is the name of the subdirectory
        class_name = os.path.basename(subdir)

        try:
            loader = None
            if filename.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif filename.lower().endswith('.pptx'):
                loader = UnstructuredPowerPointLoader(file_path)
            elif filename.lower().endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            
            if loader:
                print(f" > Processing {filename} from class '{class_name}'...")
                documents = loader.load()
                
                # Add class_name to each document's metadata
                for doc in documents:
                    doc.metadata['class'] = class_name
                
                chunks = text_splitter.split_documents(documents)
                all_chunks.extend(chunks)
                print(f"   ...found {len(chunks)} chunks.")
            else:
                print(f" > Skipping unsupported file: {filename}")
                
        except Exception as e:
            print(f"   Error processing {filename}: {e}")

print(f"\n✅ Total chunks created from all documents: {len(all_chunks)}")

# --- 3. EMBED & STORE (with updated batching logic) ---
if all_chunks:
    print("\nGenerating embeddings...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    documents_text = [chunk.page_content for chunk in all_chunks]
    
    start_time = time.time()
    embeddings = embedding_model.encode(documents_text, show_progress_bar=True)
    end_time = time.time()
    
    print(f"✅ Embeddings generated in {end_time - start_time:.2f} seconds.")

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
        {
            "class": chunk.metadata.get("class", "Unknown"),
            "source": chunk.metadata.get("source", "").split('/')[-1], 
            "page": chunk.metadata.get("page", 0)
        } 
        for chunk in all_chunks
    ]
    
    print("Adding embeddings to the collection in batches...")
    batch_size = 1000
    for i in range(0, len(all_chunks), batch_size):
        end_index = min(i + batch_size, len(all_chunks))
        collection.add(
            ids=ids[i:end_index],
            embeddings=embeddings[i:end_index].tolist(),
            documents=documents_text[i:end_index],
            metadatas=metadatas[i:end_index]
        )
        print(f"   ...added batch {i//batch_size + 1} ({end_index}/{len(all_chunks)})")

    print("\n🎉 Success! Your documents have been embedded and stored.")
    print(f"Total documents in collection: {collection.count()}")
else:
    print("\nNo documents were processed.")