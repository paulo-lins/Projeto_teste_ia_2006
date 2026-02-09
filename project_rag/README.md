# 🤖 RAG Hands-On: Arquitetura e Otimização de Busca Semântica

Este repositório foi desenvolvido para fins educacionais, servindo como um guia prático para estudantes que desejam entender o funcionamento de sistemas **RAG (Retrieval-Augmented Generation)**. 

O projeto demonstra como transformar documentos estáticos (.txt) em uma base de conhecimento dinâmica para LLMs (Modelos de Linguagem de Grande Escala), permitindo que a IA responda perguntas baseada em dados locais e privados.

---

## 🎯 Visão Geral do Projeto

A técnica de RAG resolve um dos maiores problemas das IAs atuais: a "alucinação" e o desconhecimento de dados recentes ou privados. Aqui, implementamos um pipeline completo que "ensina" a IA a consultar arquivos antes de formular uma resposta.

---

## 📂 Evolução e Diferenciais das Versões

O projeto está estruturado em duas etapas que mostram a evolução de um desenvolvedor:

### 1. Versão 1: Implementação Base (`estudo_ia_v1.py`)
* **Objetivo:** Validar o fluxo de ponta a ponta (Ingestão -> Embedding -> Busca -> Resposta).
* **Características:** Comentários em inglês para familiarização com o vocabulário técnico global.
* **Limitação:** Processa todos os documentos toda vez que é executado, gerando redundância.

### 2. Versão 2: Otimização e Performance (`estudo_ia_v2.py`)
* **Melhoria Técnica:** Implementação de lógica de **Idempotência**.
* **O que mudou:** O script agora utiliza uma verificação de existência (`any()`) para checar se cada arquivo já possui "chunks" (pedaços) vetorizados no banco local (`vector_db`).
* **Valor:** Reduz drasticamente o uso de CPU/GPU e o tempo de execução, simulando um ambiente de produção real onde performance é custo.

---

## 🛠️ Stack Tecnológica e Decisões de Projeto

| Ferramenta | Papel no Sistema | Justificativa |
| :--- | :--- | :--- |
| **Python 3.13** | Linguagem Base | Líder em ecossistemas de IA e processamento de dados. |
| **Sentence-Transformers** | Embeddings | Modelo `all-MiniLM-L6-v2`: leve (roda em CPU) e eficiente para parágrafos. |
| **OpenAI SDK** | Interface Universal | Usado como "ponte" para a Groq. Seguir este padrão permite trocar de provedor (OpenAI, Ollama, Anthropic) mudando apenas o `base_url`. |
| **Groq (Llama 3.3)** | Motor de Inferência | Provedor que oferece velocidade extrema (LPUs) e modelos Open Source de alta performance. |
| **Pickle (.pkl)** | Banco de Dados Local | Persistência binária dos vetores, facilitando o estudo sem a complexidade de um banco externo. |

> **Nota Técnica:** Embora o código utilize `import openai`, estamos conectando à API da **Groq**. Fizemos isso para seguir o padrão de mercado (OpenAI-compatible API), o que torna o código flexível para futuros provedores.

---

## 🚀 Como Configurar e Rodar

### 1. Estrutura de Pastas
```text
project_rag/
├── src/
│   ├── estudo_ia_v1.py
│   └── estudo_ia_v2.py
├── my_documents/
│   ├── files_txt/        # Coloque seus arquivos .txt aqui
│   └── vector_db/         # Gerado automaticamente (.pkl)
├── .env                  # Chave de API (GROQ_API_KEY)
├── .gitignore            # Ignora .env, .pkl e venv
└── requirements.txt