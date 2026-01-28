
#%%
#1 Função "Motor"melhorada
def processador_de_ia(texto):
    #simula a IA gerando um vetor de 3 números
    #e coloca em letra maiúscula para vermos a mudança
    vetor_simulado = [0.1, 0.2, 0.3]
    return f"VETOR: {vetor_simulado} | DOC: {texto.upper()}"

# 2. Uma lista de documentos (Simulando várias páginas de um PDF)
documentos = [
    "Receita do primeiro trimestre",
    "Custos de fabricação do iPhone",
    "Investimentos em Inteligência Artificial",
    "Projeções para 2026"
]

# Loop for
print("-----iniciando processamento-------")

count = 0
for doc in documentos:
    count = count + 1
    resultado = processador_de_ia(doc)
    print(f"Processado {count} com sucesso: {resultado}")
    




# %%
