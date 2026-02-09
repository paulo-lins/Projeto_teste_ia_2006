
#%%
import os
import pickle
import textwrap
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# 1. SETUP DE AMBIENTE
load_dotenv()

# Carrega o modelo de tradução de texto para vetores (Embeddings)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Configura o portal de comunicação com a Groq via padrão OpenAI
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

# Definição das pastas de trabalho
source_dir = os.path.join("my_documents")
vector_db_dir = os.path.join(source_dir, 'vector_db')
text_db_dir = os.path.join(source_dir, 'files_txt')

# 2. INGESTÃO E VETORIZAÇÃO (O "Cérebro" do Banco de Dados)
if not os.path.exists(vector_db_dir) or len(os.listdir(vector_db_dir)) == 0:
    os.makedirs(vector_db_dir, exist_ok=True)
    
    text_files = [f for f in os.listdir(text_db_dir) if f.endswith('.txt')]
    
    for file_name in text_files:
        # Check if ANY chunks of this file already exist in the vector database 
        # If not, we vectorize it.
        file_path = os.path.join(text_db_dir, file_name)
        already_exist = any(file_name in f for f in os.listdir(vector_db_dir))
        
        # Process only a new files
        # Vectorizing new file
        if not already_exist:
             
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # CHUNKING: Divide o texto por parágrafos duplos (\n\n)
            # Isso é vital para a IA não se perder em textos gigantes.
            chunks = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            for i, chunk_text in enumerate(chunks):
                # Transforma o texto em uma lista de números (Vetor)
                text_embedding = model.encode(chunk_text)
                
                data_to_save = {
                    "source_document": file_name,
                    "text_content": chunk_text,
                    "embedding": text_embedding
                }
                
                # Salva o dicionário em um arquivo binário (.pkl)
                pkl_name = f"{file_name}_chunk_{i+1}.pkl"
                pkl_path = os.path.join(vector_db_dir, pkl_name)
                
                with open(pkl_path, 'wb') as pkl_file:
                    pickle.dump(data_to_save, pkl_file)
                    
    print("Base de conhecimento local atualizada com sucesso.")

# 3. BUSCA SEMÂNTICA (O "Filtro" Inteligente)
user_query = "Qual o faturamento da apple em 2026"
query_embedding = model.encode(user_query)
search_results = []

for pkl_file in os.listdir(vector_db_dir):
    if pkl_file.endswith('.pkl'):
        pkl_path = os.path.join(vector_db_dir, pkl_file)
        with open(pkl_path, 'rb') as f:
            stored_data = pickle.load(f)
            
            # Calcula a similaridade matemática entre a pergunta e o texto salvo
            score = util.cos_sim(query_embedding, stored_data['embedding'])
            
            search_results.append({
                "text_content": stored_data['text_content'],
                "score": score.item()
            })

# Ordena do mais relevante para o menos relevante
search_results.sort(key=lambda x: x['score'], reverse=True)
context_text = "\n".join([f"- {item['text_content']}" for item in search_results[:3]])

# 4. RESPOSTA FINAL (A "Voz" da IA)
completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": f"Você é um analista sênior. Responda em Português usando: {context_text}"
        },
        {"role": "user", "content": user_query}
    ],
    temperature=0
)

print(f"\nAnálise concluída:\n")
print(textwrap.fill(completion.choices[0].message.content, width=75))
# %%
