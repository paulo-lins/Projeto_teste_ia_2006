#%%
#from sentence_transformers import SentenceTransformer
from sentence_transformers import util

# 1. Vamos carregar o modelo (o "cérebro")
print("Carregando o modelo...")
# model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Dados de estudo (vamos imaginar que são trechos em PDF)
# textos = [
#   "A inteligência artificial não é sobre substituir o humano, mas sobre potencializar a capacidade humana de resolver problemas complexos.",
#   "A IA é a eletricidade do século XXI: ela vai transformar todas as indústrias, da saúde à educação.",
#   "O maior perigo da IA não é que ela se torne consciente, mas que ela seja incrivelmente eficiente em objetivos mal alinhados com os nossos.",
#   "Inteligência artificial é a arte de fazer as máquinas agirem de forma que chamaríamos de inteligente se um ser humano fizesse o mesmo.",
#   "A IA nos permite delegar o processamento de dados para que possamos focar no que é essencialmente humano: a criatividade e a empatia.",
#   "Ferramentas de IA são como espelhos; elas refletem os dados que fornecemos, incluindo nossas descobertas e nossos preconceitos.",
#   "Em um mundo movido por algoritmos, a curiosidade humana e a capacidade de fazer as perguntas certas tornam-se o nosso maior diferencial.",
#   "A verdadeira revolução da IA acontece quando ela se torna invisível, integrada de forma fluida no nosso cotidiano.",
#   "A inteligência artificial pode prever o futuro com base em padrões, mas a vontade humana ainda é o que o constrói.",
#   "A questão não é se as máquinas podem pensar, mas sim como podemos pensar melhor ao lado delas."
# ]

# loop for para processar e mostrar a matematica por tras do texto
# for frase in textos:
#     vetor = model.encode(frase)
#     print(f"\nFrase {frase}")
#     print(f"Dimensòes do Vetor {len(vetor)}")
#     print(f"3 primeiros números do vetor {vetor[:3]}")
    
# 1. Geramos os embeddings das 3 frases
v1 = model.encode("A inteligência artificial aprende com dados.")
v2 = model.encode("O mercado financeiro usa IA para previsões.")
v3 = model.encode("Hoje o dia está ensolarado e vou à praia.")
 
# 2. Compara os embeddings das frases 1 e 2
sim_1_2 = util.cos_sim(v1, v2)

# compara 1 e 3
sim_1_3 = util.cos_sim(v1, v3)

print(f"Similaridade IA vs Finanças: {sim_1_2.item():.4f}")
print(f"Similaridade IA vs Praia: {sim_1_3.item():.4f}")

if sim_1_2 > sim_1_3:
    print("os vetores 1 e 2 sao mais similares")
else:
    print("os vetores 2 e 3 sao mais similares")

# %%
