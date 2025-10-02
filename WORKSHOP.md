# Workshop — Criando um Copilot Personalizado com RAG (Gemini + FAISS)

> **Duração sugerida:** 90–120 min

Este workshop conduz, passo a passo, a construção de um **copilot** que responde perguntas
**com base nas Boas Práticas da sua empresa** e **no seu código-fonte**, usando **RAG** (Retrieval-Augmented Generation),
**Gemini (Google)** para *embeddings* e *geração*, e **FAISS** para busca vetorial.

---

## Objetivos de Aprendizagem

Ao final, o aluno será capaz de:

1. Explicar o fluxo **RAG** (ingestão ➜ indexação ➜ recuperação ➜ geração).
2. Preparar dados (texto e código) e **chunkar** conteúdo com sobreposição.
3. Gerar **embeddings** com `gemini-embedding-001`.
4. Construir e consultar um índice **FAISS** (similaridade cosseno via produto interno em vetores normalizados).
5. Montar *prompts* que **citam o contexto recuperado** para o modelo de geração (`gemini-2.5-flash`).
6. Executar um **chat RAG** que “cita” arquivos do código nos trechos relevantes.

---

## Pré-requisitos

- **Python 3.11+**  
- **Conta Google** com chave **GEMINI_API_KEY** (gratuita para devs/estudantes)
- Conhecimentos básicos de:
  - Terminal/CLI
  - Python (venv, pacotes)
  - Leitura de código

---

## Setup do Ambiente (10–15 min)

> Execute os passos **dentro da pasta** `genai-copilot-lab/`.

1) **Crie o ambiente virtual e instale dependências**
~~~bash
# opção com pyenv-virtualenv
pyenv virtualenv 3.11.11 genai-copilot-lab
pyenv local genai-copilot-lab

# ou venv nativo
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
~~~

2) **Configure o `.env`**  
Crie um arquivo `.env` com o conteúdo:

~~~dotenv
GEMINI_API_KEY="SUA_CHAVE_AQUI"
# opcionais (defaults já definidos no código)
# GEMINI_CHAT_MODEL="gemini-2.5-flash"
# GEMINI_EMBED_MODEL="gemini-embedding-001"
~~~

> **Dica:** Se quiser, crie um `.env.example` para os alunos e peça para copiarem para `.env`.

---

## Estrutura do Projeto (resumo)

~~~text
genai-copilot-lab/
  ├─ main.py                # CLI do Copilot com RAG (Gemini + FAISS)
  ├─ README.md
  ├─ WORKSHOP.md            # (este arquivo)
  ├─ requirements.txt
  ├─ .gitignore
  └─ (artefatos gerados em runtime)
      bp_faiss.index
      bp_chunks.pkl
      code_faiss.index
      code_chunks.pkl
~~~

---

## Roteiro Prático (mão na massa)

### Passo 0 — Sanidade do SDK (opcional, 3 min)

> Se quiser testar rapidamente sua chave antes do RAG:

~~~bash
python - << 'PY'
from google import genai
import os
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
r = client.models.generate_content(model="gemini-2.5-flash", contents="Diga 'ok Gemini' em pt-br.")
print(r.text)
PY
~~~

Se imprimir algo como “ok Gemini”, estamos prontos.

---

### Passo 1 — Ingestão de **Boas Práticas** (10–12 min)

1) Crie uma pasta `dados/` e um arquivo `boas_praticas.txt` (ou use o seu documento real):

~~~bash
mkdir -p dados
~~~

Abra `dados/boas_praticas.txt` e cole algo como:

~~~text
# Boas Práticas — Equipe XPTO
1. Logging deve usar níveis apropriados (INFO, WARNING, ERROR).
2. Funções devem ter docstrings claras e tipagem estática sempre que possível.
3. Evitar duplicação de código (DRY). Refatorar funções muito longas.
4. Testes unitários para funções críticas; cobertura mínima recomendada: 80%.
5. Configurações sensíveis via variáveis de ambiente (.env), nunca em código.
~~~

2) Ingerir o `.txt` no índice FAISS:

~~~bash
python main.py ingest-best --file ./dados/boas_praticas.txt
~~~

> Saída esperada: criação de `bp_faiss.index` e `bp_chunks.pkl`.

---

### Passo 2 — Ingestão de **Código-Fonte** (12–15 min)

Você pode usar **qualquer repositório pequeno** ou, para fins de aula, apontar para `genai-tests-lab/` (já presente no monorepo):

~~~bash
# exemplo: ingerir apenas .py e .md (padrão)
python main.py ingest-code --dir ../genai-tests-lab

# (opcional) especificar extensões:
python main.py ingest-code --dir ../genai-tests-lab --ext ".py,.md,.txt"
~~~

> Saída esperada: criação de `code_faiss.index` e `code_chunks.pkl`.  
> Os *chunks* de código são prefixados com `"[FILE]: caminho/arquivo.py"` para que o modelo possa referenciar o arquivo na resposta.

---

### Passo 3 — **Chat** RAG (15–20 min)

Inicie o chat interativo:

~~~bash
python main.py chat
~~~

Faça perguntas que conectem **regras** às **partes do código**:

~~~text
>> Nosso logging está de acordo com as boas práticas?
>> Onde há funções longas que valem refatoração?
>> Quais arquivos mostram ausência de docstrings?
>> Como garantir 80%+ de cobertura nos módulos principais?
~~~

> **Como o modelo responde?**  
> - Ele gera respostas **com base ESTRITA** no contexto recuperado.  
> - Ao citar trechos de código, ele deve mencionar `[FILE]: ...` com o nome do arquivo que veio do chunk.

Para sair, digite `sair`.

---

### Passo 4 — Pergunta Única (modo rápido, 2–3 min)

~~~bash
python main.py ask --question "O módulo sum.py segue nossas boas práticas?"
~~~

Útil para pipelines ou consultas avulsas.

---

## Entendendo o que acontece por trás

### 1) Chunking
- Os textos (boas práticas) e arquivos de código são **quebrados em janelas** (~1200 chars, *overlap* 200) para melhorar a recuperação.
- Cada chunk de código recebe um prefixo:  
  `"[FILE]: relativo/para/o/diretorio\n<conteudo do arquivo>"`

### 2) Embeddings
- Cada *chunk* vira um vetor com o modelo `gemini-embedding-001`.
- Todos os vetores são **normalizados** (L2=1) para usar **IndexFlatIP** (produto interno ≈ cosseno).

### 3) FAISS
- Criamos dois índices: **Boas Práticas** e **Código**.
- Na consulta, transformamos a pergunta em embedding e **recuperamos k** vizinhos de cada índice.

### 4) Geração (LLM)
- Montamos um *prompt* com:
  - **Contexto de Boas Práticas** (top-k)
  - **Contexto de Código** (top-k)
  - **Pergunta do usuário**
- O modelo `gemini-2.5-flash` responde **com base nesses trechos**.
- Se algo **não** estiver no contexto, o modelo é instruído a **dizer que não sabe**.

---

## Desafios (para além do básico)

1) **Tunar k**  
   - No código (`retrieve_contexts`), mude `k_bp`/`k_code` e observe a qualidade das respostas.  
   - Projete perguntas onde **mais contexto** melhora a resposta (e onde **atrapalha**).

2) **Citações marcadas**  
   - Alterar `generate_answer`/`build_prompt` para **enumerar** os chunks usados e **incluir um “Referências:”** no fim da resposta.

3) **Filtros por extensão**  
   - Exponha `--ext` de forma mais rica: por exemplo, `.py` recebe *chunking* diferente de `.md`.

4) **Paralelizar embeddings**  
   - (Avançado) Faça *batching* de até N textos por chamada de `embed_content` (se o SDK oferecer).

5) **Interface Web**  
   - Crie um *frontend* mínimo (Flask/FastAPI + HTMX) que conversa com as mesmas funções.

---

## Troubleshooting (FAQ rápido)

- **`GEMINI_API_KEY` faltando**  
  > Verifique `.env`. Rode `python -c "import os;print(bool(os.getenv('GEMINI_API_KEY')))"`.

- **`faiss` não instala**  
  > Use `faiss-cpu` (já no `requirements.txt`). Em alguns ambientes Windows, é mais simples usar WSL/conda.

- **`RuntimeError: Formato inesperado de retorno de embedding`**  
  > Atualize `google-genai` (`pip install -U google-genai`). O código tenta lidar com variações de estrutura; se mudar novamente, ajuste `extract_embedding`.

- **Dimensão inconsistente no índice**  
  > Remova artefatos e reingira:  
  > `python main.py reset` ➜ `ingest-best` ➜ `ingest-code`.

- **Respostas “alucinadas”**  
  > O prompt instrui a dizer “não sei”. Se persistir, reduza `k`, melhore as práticas no `.txt`, ou torne as perguntas mais específicas.

---

## Glossário Rápido

- **RAG:** Recupera contexto relevante do seu *corpus* (base própria) e injeta no *prompt* antes de gerar a resposta.
- **Embedding:** Vetor numérico que “representa” um texto; textos parecidos → vetores próximos.
- **FAISS:** Biblioteca de busca vetorial eficiente (Facebook AI Similarity Search).
- **Chunk:** Pedaço de texto/código menor para indexação e recuperação.

