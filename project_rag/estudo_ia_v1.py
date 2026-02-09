#%%
import os
import pickle
import textwrap
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
load_dotenv()
print(f"DEBUG: Groq Key carregada? {'Sim' if os.getenv('GROQ_API_KEY') else 'Não'}")

# 1. ENVIRONMENT SETUP
load_dotenv()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Configuring the OpenAI client to connect to the Groq API
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

# --- FILE STRUCTURE CONFIGURATION ---
base_path = "my_documents"
text_docs_dir = os.path.join(base_path, 'files_txt') # Source .txt files
vector_db_dir = os.path.join(base_path, 'vector_db')    # Destination .pkl files

# source_dir = os.path.join("my_documents")
# vector_db_dir = os.path.join(source_dir, 'vector_db')
# text_db_dir = os.path.join(source_dir, 'files_txt')


# Ensure directories exist
os.makedirs(text_docs_dir, exist_ok=True)
os.makedirs(vector_db_dir, exist_ok=True)

print(f"🔍 Checking document folder: {os.path.abspath(text_docs_dir)}")

# 2. DATA INGESTION & VECTORIZATION
# Identify all text files in the source directory
text_files = [f for f in os.listdir(text_docs_dir) if f.endswith('.txt')]

if not text_files:
    print(f"⚠️ WARNING: No .txt files found in {text_docs_dir}!")

for file_name in text_files:
    # Check if this file has already been vectorized to avoid redundant work
    is_already_processed = any(file_name in f for f in os.listdir(vector_db_dir))
    
    if not is_already_processed:
        print(f"🔄 Vectorizing new document: {file_name}...")
        file_path = os.path.join(text_docs_dir, file_name)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Split text into chunks by double newlines (paragraphs)
            chunks = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            for i, chunk_text in enumerate(chunks):
                # Generate the numerical representation (embedding) of the text
                text_vector = embedding_model.encode(chunk_text)
                
                payload = {
                    "source_document": file_name,
                    "text_content": chunk_text,
                    "embedding": text_vector
                }
                
                # Save as a binary pickle file
                pkl_filename = f"{file_name}_chunk_{i+1}.pkl"
                pkl_path = os.path.join(vector_db_dir, pkl_filename)
                
                with open(pkl_path, 'wb') as pkl_file:
                    pickle.dump(payload, pkl_file)
        print(f"✅ {file_name} processed and stored.")

# 3. SEMANTIC SEARCH (Retrieval)
user_query = "What are Apple's plans for 2026?"
query_vector = embedding_model.encode(user_query)
search_results = []

# Verify if the vector database is not empty
if os.path.exists(vector_db_dir) and os.listdir(vector_db_dir):
    for pkl_file in os.listdir(vector_db_dir):
        if pkl_file.endswith('.pkl'):
            pkl_path = os.path.join(vector_db_dir, pkl_file)
            with open(pkl_path, 'rb') as f:
                stored_data = pickle.load(f)
                # Calculate mathematical similarity (Cosine Similarity)
                similarity_score = util.cos_sim(query_vector, stored_data['embedding'])
                
                search_results.append({
                    "text_content": stored_data['text_content'],
                    "score": similarity_score.item()
                })
    # Sort results by the highest similarity score
    search_results.sort(key=lambda x: x['score'], reverse=True)
else:
    print("❌ Vector database is empty.")

# 4. FINAL LLM RESPONSE (Augmentation & Generation)
# Combine the top 3 relevant chunks into a single context string
if search_results:
    context_text = "\n".join([f"- {item['text_content']}" for item in search_results[:3]])
else:
    context_text = "NO LOCAL DATA FOUND."

completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system", 
            "content": f"Use ONLY the following context to answer the user. If the context is empty, state that you cannot help without local files.\n\nCONTEXT:\n{context_text}"
        },
        {"role": "user", "content": user_query}
    ],
    temperature=0
)

print(f"\n✅ Paulo, analysis complete:\n")
print(textwrap.fill(completion.choices[0].message.content, width=75))
# %%
