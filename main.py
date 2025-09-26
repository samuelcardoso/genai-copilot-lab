#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copilot Personalizado com RAG (Gemini + FAISS)
----------------------------------------------

Funcionalidades (CLI):
  1) Ingestão de Boas Práticas (arquivo .txt)
     $ python main.py ingest-best --file ./dados/boas_praticas.txt

  2) Ingestão de Código (diretório com .py, .md, etc.)
     $ python main.py ingest-code --dir ../meu_projeto

  3) Chat (interativo) usando RAG sobre Best Practices + Código
     $ python main.py chat
     >> sua pergunta aqui
     >> sair  (para encerrar)

  4) Pergunta única (não interativa)
     $ python main.py ask --question "como melhorar a função X?"

  5) Reset (apaga índices/artefatos locais)
     $ python main.py reset

Requisitos:
  - GEMINI_API_KEY no arquivo .env

Observação:
  - Utiliza FAISS (IndexFlatIP com vetores normalizados) para aproximar similaridade cosseno.
  - Salva índices e chunks em arquivos no próprio diretório do projeto.
"""

import argparse
import os
import sys
import re
import pickle
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import numpy as np
import faiss
from dotenv import load_dotenv
from google import genai

# =========================
# Configurações do Modelo
# =========================
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

# =========================
# Artefatos / Caminhos
# =========================
ARTIFACT_DIR = Path(__file__).parent.resolve()

# Best Practices
BP_CHUNKS_PKL = ARTIFACT_DIR / "bp_chunks.pkl"
BP_INDEX_FILE = ARTIFACT_DIR / "bp_faiss.index"

# Código
CODE_CHUNKS_PKL = ARTIFACT_DIR / "code_chunks.pkl"
CODE_INDEX_FILE = ARTIFACT_DIR / "code_faiss.index"

# =========================
# Utilidades gerais
# =========================

def load_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERRO: defina GEMINI_API_KEY no seu arquivo .env", file=sys.stderr)
        sys.exit(1)
    return genai.Client(api_key=api_key)

def normalize(vecs: np.ndarray) -> np.ndarray:
    """Normaliza cada vetor para norma L2 = 1 (p/ similaridade cosseno via IP)."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms

def new_ip_index(dimension: int) -> faiss.IndexFlatIP:
    return faiss.IndexFlatIP(dimension)

def save_index_and_chunks(index: faiss.Index, chunks: List[str], index_path: Path, chunks_path: Path):
    faiss.write_index(index, str(index_path))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

def load_index_and_chunks(index_path: Path, chunks_path: Path) -> Tuple[faiss.Index, List[str]]:
    if not (index_path.exists() and chunks_path.exists()):
        return None, None
    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def delete_artifacts():
    for p in [BP_CHUNKS_PKL, BP_INDEX_FILE, CODE_CHUNKS_PKL, CODE_INDEX_FILE]:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

# =========================
# Chunking
# =========================

def split_text_by_chars(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """
    Quebra texto em janelas de caracteres com sobreposição.
    Útil para .txt de boas práticas.
    """
    text = re.sub(r"\s+\n", "\n", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]

def chunk_code(content: str, filename: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """
    Quebra código em janelas; prefixa com o nome do arquivo para contexto.
    """
    header = f"[FILE]: {filename}\n"
    raw = header + content
    return [c for c in split_text_by_chars(raw, max_chars, overlap)]

def iter_code_files(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    exts = {e.lower() for e in exts}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

# =========================
# Embeddings (Gemini)
# =========================

def extract_embedding(resp) -> List[float]:
    """
    Tenta extrair o vetor de embedding do retorno do google-genai,
    acomodando variações de estrutura.
    """
    # Formato comum: resp.embeddings -> [floats] ou [obj(values=list)]
    emb = getattr(resp, "embeddings", None)
    if emb is None:
        emb = getattr(resp, "embedding", None)
        if emb is not None and hasattr(emb, "values"):
            return list(emb.values)

    if isinstance(emb, list):
        first = emb[0] if emb else None
        if isinstance(first, list):
            return list(first)
        if hasattr(first, "values"):
            return list(first.values)
        if hasattr(first, "embedding"):
            return list(first.embedding)

        # Às vezes resp.embeddings já é a lista de floats:
        if all(isinstance(x, (int, float)) for x in emb):
            return [float(x) for x in emb]

    # Fallback hard: tente acessar .values direto
    if hasattr(resp, "values"):
        return list(resp.values)

    raise RuntimeError("Formato inesperado de retorno de embedding do Gemini.")

def embed_texts(client: genai.Client, texts: List[str]) -> np.ndarray:
    """
    Gera embeddings para uma lista de textos (serialmente para simplicidade/estabilidade).
    Retorna ndarray float32 [N, D].
    """
    vectors = []
    for t in texts:
        r = client.models.embed_content(model=EMBED_MODEL, contents=t)
        vec = extract_embedding(r)
        vectors.append(vec)
    arr = np.array(vectors, dtype="float32")
    return arr

# =========================
# Ingestão
# =========================

def ingest_best_practices(client: genai.Client, txt_path: Path):
    if not txt_path.exists():
        print(f"ERRO: arquivo não encontrado: {txt_path}", file=sys.stderr)
        sys.exit(2)

    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    chunks = split_text_by_chars(text, max_chars=1200, overlap=200)

    if not chunks:
        print("Nenhum chunk gerado a partir do .txt. Verifique o arquivo.", file=sys.stderr)
        sys.exit(3)

    print(f"> Gerando embeddings para {len(chunks)} chunks de boas práticas...")
    vecs = embed_texts(client, chunks)
    vecs = normalize(vecs)

    index = new_ip_index(vecs.shape[1])
    index.add(vecs)

    save_index_and_chunks(index, chunks, BP_INDEX_FILE, BP_CHUNKS_PKL)
    print(f"OK! Índice de boas práticas salvo: {BP_INDEX_FILE.name} ({len(chunks)} chunks)")

def ingest_codebase(client: genai.Client, dir_path: Path, extensions: List[str]):
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"ERRO: diretório inválido: {dir_path}", file=sys.stderr)
        sys.exit(2)

    files = list(iter_code_files(dir_path, extensions))
    if not files:
        print(f"Nenhum arquivo com extensões {extensions} encontrado em {dir_path}.", file=sys.stderr)
        sys.exit(3)

    chunks = []
    for fp in files:
        try:
            content = fp.read_text(encoding="utf-8", errors="ignore")
            parts = chunk_code(content, filename=str(fp.relative_to(dir_path)))
            chunks.extend(parts)
        except Exception:
            # Pula arquivos que não puderem ser lidos
            continue

    if not chunks:
        print("Nenhum chunk de código foi gerado. Verifique o diretório.", file=sys.stderr)
        sys.exit(4)

    print(f"> Gerando embeddings para {len(chunks)} chunks de código...")
    vecs = embed_texts(client, chunks)
    vecs = normalize(vecs)

    index = new_ip_index(vecs.shape[1])
    index.add(vecs)

    save_index_and_chunks(index, chunks, CODE_INDEX_FILE, CODE_CHUNKS_PKL)
    print(f"OK! Índice de código salvo: {CODE_INDEX_FILE.name} ({len(chunks)} chunks)")

# =========================
# Busca (RAG)
# =========================

def search(index: faiss.Index, query_vec: np.ndarray, k: int = 5) -> Tuple[List[int], List[float]]:
    """Retorna (indices, scores) dos k vizinhos mais próximos."""
    D, I = index.search(query_vec, k)
    return I[0].tolist(), D[0].tolist()

def retrieve_contexts(client: genai.Client, question: str, k_bp: int = 4, k_code: int = 4) -> Dict[str, List[str]]:
    # Carrega índices/CHUNKS se existirem
    bp_index, bp_chunks = load_index_and_chunks(BP_INDEX_FILE, BP_CHUNKS_PKL)
    code_index, code_chunks = load_index_and_chunks(CODE_INDEX_FILE, CODE_CHUNKS_PKL)

    # Embedding da pergunta
    q_vec = embed_texts(client, [question])
    q_vec = normalize(q_vec)

    contexts = {"best_practices": [], "code": []}

    if bp_index and bp_chunks:
        I, _ = search(bp_index, q_vec, k_bp)
        contexts["best_practices"] = [bp_chunks[i] for i in I if i < len(bp_chunks)]

    if code_index and code_chunks:
        I, _ = search(code_index, q_vec, k_code)
        contexts["code"] = [code_chunks[i] for i in I if i < len(code_chunks)]

    return contexts

# =========================
# Geração (Gemini)
# =========================

def build_prompt(question: str, contexts: Dict[str, List[str]]) -> str:
    bp = "\n\n---\n".join(contexts.get("best_practices", [])[:4]) or "(nenhum contexto de boas práticas disponível)"
    cd = "\n\n---\n".join(contexts.get("code", [])[:4]) or "(nenhum contexto de código disponível)"

    system = (
        "Você é um copiloto de engenharia de software que utiliza RAG.\n"
        "Responda sempre com base ESTRITA no contexto fornecido.\n"
        "Se a resposta não estiver no contexto, diga claramente que não sabe.\n"
        "Quando referenciar trechos de código, cite o arquivo em [FILE]: nome_do_arquivo.\n"
        "Se fizer recomendações, alinhe com as boas práticas quando houver.\n"
    )

    prompt = (
        f"{system}\n\n"
        f"# CONTEXTO — BOAS PRÁTICAS\n{bp}\n\n"
        f"# CONTEXTO — CÓDIGO\n{cd}\n\n"
        f"# PERGUNTA DO USUÁRIO\n{question}\n\n"
        "Responda em português, objetiva e didaticamente. Se útil, liste passos acionáveis."
    )
    return prompt

def generate_answer(client: genai.Client, prompt: str) -> str:
    resp = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    )
    return getattr(resp, "text", "").strip() or "(sem texto de resposta)"

# =========================
# CLI / Fluxos
# =========================

def cmd_ingest_best(args):
    client = load_client()
    ingest_best_practices(client, Path(args.file))

def cmd_ingest_code(args):
    client = load_client()
    exts = args.ext or ".py,.md"
    extensions = [e.strip() if e.strip().startswith(".") else f".{e.strip()}" for e in exts.split(",")]
    ingest_codebase(client, Path(args.dir), extensions)

def cmd_chat(_args):
    client = load_client()
    print("Chat RAG — digite sua pergunta (ou 'sair' para encerrar).")
    while True:
        try:
            q = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"sair", "exit", "quit"}:
            break
        ctx = retrieve_contexts(client, q, k_bp=4, k_code=4)
        prompt = build_prompt(q, ctx)
        answer = generate_answer(client, prompt)
        print("\n=== Resposta ===\n" + answer + "\n")

def cmd_ask(args):
    client = load_client()
    q = args.question.strip()
    ctx = retrieve_contexts(client, q, k_bp=4, k_code=4)
    prompt = build_prompt(q, ctx)
    answer = generate_answer(client, prompt)
    print(answer)

def cmd_reset(_args):
    delete_artifacts()
    print("Artefatos removidos (índices e pkl).")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="genai-copilot-lab",
        description="Criando um Copilot Personalizado com RAG (Gemini + FAISS)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("ingest-best", help="Ingerir .txt de boas práticas")
    pb.add_argument("--file", required=True, help="Caminho para o .txt de boas práticas")
    pb.set_defaults(func=cmd_ingest_best)

    pc = sub.add_parser("ingest-code", help="Ingerir diretório de código")
    pc.add_argument("--dir", required=True, help="Diretório raiz do código")
    pc.add_argument("--ext", default=".py,.md", help="Extensões separadas por vírgula (ex.: .py,.md,.txt)")
    pc.set_defaults(func=cmd_ingest_code)

    ch = sub.add_parser("chat", help="Inicia chat interativo com RAG")
    ch.set_defaults(func=cmd_chat)

    ask = sub.add_parser("ask", help="Faz uma pergunta única (não interativa)")
    ask.add_argument("--question", required=True, help="Pergunta em texto")
    ask.set_defaults(func=cmd_ask)

    rs = sub.add_parser("reset", help="Remove índices/artefatos gerados")
    rs.set_defaults(func=cmd_reset)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
