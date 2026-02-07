# Apple Intelligence RAG - Analisador Semântico - Paulo Lins

Sistema de Retrieval-Augmented Generation (RAG) de alta performance, desenvolvido para análise de relatórios financeiros e técnicos da Apple. O projeto integra a velocidade de inferência da Groq com o processamento local do chip Apple M4.

## Funcionalidades
- Busca Semântica: Processamento de linguagem natural para compreensão de contexto além de palavras-chave.
- Processamento Local: Geração de embeddings localmente via SentenceTransformers (modelo all-MiniLM-L6-v2).
- Inferência de Baixa Latência: Integração com a API do Groq utilizando o modelo Llama 3.3 70B.
- Persistência de Dados: Armazenamento vetorial otimizado via serialização pickle.

## Especificações de Hardware
- Processador: Apple M4 (MacBook Pro / Mac Mini / iMac)
- Otimização: Aproveitamento da arquitetura do chip para operações de tensores e geração de vetores.



## Pré-requisitos
É necessário possuir uma conta na Groq Cloud para obtenção da chave de API.

### Instalação e Configuração

1. Clonar o repositório:
   ```bash
   git clone [https://github.com/seu-usuario/projeto-rag-apple.git](https://github.com/seu-usuario/projeto-rag-apple.git)
   cd projeto-rag-apple                     