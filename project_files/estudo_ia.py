# O Pipeline da Busca (Retrieval)
# Sua Pergunta: "O que a Apple planeja para 2026?"

# Vetorização da Pergunta: O modelo transforma sua 
# frase em um vetor (chamaremos de vetor_pergunta).

# Comparação: O Python abre cada arquivo .pkl na 
# pasta BD e compara o vetor_pergunta com o vetor_trecho salvo.

# Ranking: Ele calcula a similaridade (quão parecidos os números são) 
# e te mostra o texto que teve a maior nota.

# importando bibliotecas e frameworks
#%%
import os
from dotenv import load_dotenv
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util
from groq import Groq
import textwrap

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')

# Defining the text file and database folders
source_dir = os.path.join("my_documents")
vector_db_dir = os.path.join(source_dir, 'vector_db')
text_db_dir = os.path.join(source_dir, 'files_txt')

# TXT to PKL (Data ingestion)
if not os.path.exists(vector_db_dir) or len(os.listdir(vector_db_dir))==0:

    os.makedirs(vector_db_dir, exist_ok = True)

# list all TXT files with filter for only .txt files
text_files = [f for f in os.listdir(text_db_dir) if f.endswith('.txt')]
# Data ingestion (TXT to PKL)
for file_name in text_files:
    # acces the complete path
    file_path = os.path.join(text_db_dir, file_name)

# Abrindo os arquivos para fazer o chunking
    with open(file_path, 'r', encoding = 'utf-8') as file:
        content = file.read()
         # Chunking slicing by paragraphs
        chunks = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for i, chunk_text in enumerate(chunks):
            # Embedding
            text_embedding = model.encode(chunk_text)
            
            # Creating a pickle dictionary
            data_to_save = {
                "source_document": file_name,
                "text_content": chunk_text,
                "embedding": text_embedding
            }
            
            # Defining the file name .pkl
            pkl_name = f"{file_name}_chunk_{i+1}.pkl"
            pkl_path = os.path.join(vector_db_dir, pkl_name)
            
            # Save in local database
            with open(pkl_path, 'wb') as pkl_file:
                pickle.dump(data_to_save, pkl_file)
                
print("Data ingestion complete")

# ----- Query (user question)
user_query = "Quais são os planos da apple para 2026"
query_embedding = model.encode(user_query)
            
search_results = []

# Going through the .pkl files for comparison
for pkl_file in os.listdir(vector_db_dir):
    if pkl_file.endswith('.pkl'):
        pkl_path = os.path.join(vector_db_dir, pkl_file)
        
        # open the 'box'
        with open(pkl_path, 'rb') as f:
            # Load the complete dictionary 
            stored_data = pickle.load(f)
            
            # Here, the AI ​​measures the similarity between the question
            # and the saved text excerpt.
            score = util.cos_sim(query_embedding, stored_data['embedding'])
            
            # We store the data and the score in our list.
            search_results.append({
                "source_document": stored_data['source_document'],
                "text_content": stored_data['text_content'],
                "score": score.item()
            })
search_results.sort(key=lambda x: x['score'], reverse = True)

response = search_results[:3]

# 3. PREPARAÇÃO DO CONTEXTO
# Extrai o conteúdo de texto dos itens selecionados e os une em uma única string
# Isso será entregue ao Llama como a "base de conhecimento"
context_text = "\n".join([f"- {item['text_content']}" for item in response])

# 4. CONFIGURAÇÃO DO CLIENTE GROQ
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 5. CHAMADA AO MODELO DE LINGUAGEM (LLM)
# Enviamos a pergunta + os dados do relatório para o Llama 3.3 70B
completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": f"""Você é um analista de tecnologia sênior e amigável.
            Instruções:
            1. Responda ao usuário (Paulo) em Português.
            2. Seja direto e use um tom de conversa natural.
            3. Use APENAS as informações do contexto abaixo para responder.
            4. Se não souber a resposta diga: Desculpe, eu não sei.
            5. Responda em no máximo dois parágrafos
            
            CONTEXTO:
            {context_text}"""
        },
        {
            "role": "user",
            "content": user_query # A pergunta original que você digitou
        }
    ],
    temperature=0.7 # Equilíbrio entre precisão e fluidez natural
)

# 6. EXIBIÇÃO DO RESULTADO
answer = completion.choices[0].message.content

print(f"\n✅ Paulo, aqui está a análise sobre '{user_query}':\n")

# O fill() quebra o texto automaticamente para não passar de 80 colunas
formatted_answer = textwrap.fill(answer, width=50)

print(formatted_answer)
 
            
            

# %%

# %%
