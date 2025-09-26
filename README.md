# genai-copilot-lab — Criando um Copilot Personalizado com RAG (Gemini + FAISS)

Este lab ensina a técnica **RAG (Retrieval-Augmented Generation)** criando um “copilot” que
responde perguntas **com base nas suas Boas Práticas** e **no seu código-fonte**.

## Visão Geral

- **Embeddings** com `Gemini` → **FAISS** (busca vetorial por similaridade)
- **Consulta**: a pergunta é convertida em embedding → recupera *k* trechos de:
  - **Boas Práticas** (ingestão via `.txt`)
  - **Código-fonte** (ingestão via diretório)
- **Geração**: o modelo `Gemini` responde **somente** com base nesses trechos.

## Requisitos

- Python 3.11+ (recomendado)
- Conta Google com **GEMINI_API_KEY** (gratuita para estudantes/devs)
- Sistema com `faiss-cpu` funcional (já no `requirements.txt`)

## Setup do Ambiente

```bash
# 1) Crie e ative o venv (exemplo com pyenv-virtualenv)
pyenv virtualenv 3.11.11 genai-copilot-lab
pyenv local genai-copilot-lab

# ou: python -m venv .venv && source .venv/bin/activate

# 2) Dependências
pip install -r requirements.txt

# 3) Variáveis de ambiente
cp .env.example .env   # se preferir, crie manualmente
# edite .env e defina:
# GEMINI_API_KEY="sua_chave_aqui"
