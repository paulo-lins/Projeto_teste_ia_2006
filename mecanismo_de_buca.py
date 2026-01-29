#%%
# from sentence_transformers import SentenceTransformer, util

# model = SentenceTransformer('all-MiniLM-L6-v2')

#1. Base de dados
base_de_dados = [
    "A inteligência artificial não é sobre substituir o humano, mas sobre potencializar a capacidade humana de resolver problemas complexos.",
  "A IA é a eletricidade do século XXI: ela vai transformar todas as indústrias, da saúde à educação.",
  "O maior perigo da IA não é que ela se torne consciente, mas que ela seja incrivelmente eficiente em objetivos mal alinhados com os nossos.",
  "Inteligência artificial é a arte de fazer as máquinas agirem de forma que chamaríamos de inteligente se um ser humano fizesse o mesmo.",
  "A IA nos permite delegar o processamento de dados para que possamos focar no que é essencialmente humano: a criatividade e a empatia.",
  "Ferramentas de IA são como espelhos; elas refletem os dados que fornecemos, incluindo nossas descobertas e nossos preconceitos.",
  "Em um mundo movido por algoritmos, a curiosidade humana e a capacidade de fazer as perguntas certas tornam-se o nosso maior diferencial.",
  "A verdadeira revolução da IA acontece quando ela se torna invisível, integrada de forma fluida no nosso cotidiano.",
  "A inteligência artificial pode prever o futuro com base em padrões, mas a vontade humana ainda é o que o constrói.",
  "A questão não é se as máquinas podem pensar, mas sim como podemos pensar melhor ao lado delas."
]


#2. Query
query = "A inteligencia artificial vai prever o futuro?"

#3. Embeddings
query_embeddings = model.encode(query)
print(len(query_embeddings))

ranking = []

#4. Loop de comparação
for frase in base_de_dados:
    vetor_texto = model.encode(frase)
    score = util.cos_sim(query_embeddings, vetor_texto).item()
    ranking.append({"frase": frase, "score": score})
    
#5. Ordenando o resuldado do maior para o menor
#O parâmetro key é exatamente o critério 
# que você escolhe para a ordenação.
ranking_ordenado = sorted(ranking, key = lambda x : x['score'], reverse=True)[:3]

#6.Mostrar resultados
print(f"Mostrando resultado para: {query}")
for i, item in enumerate (ranking_ordenado, start=1):   
    print(f"\n Resposta: N. {i}, score: {item['score']:.2f}, {item['frase']}")
# %%
