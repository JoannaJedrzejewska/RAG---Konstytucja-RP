import json, os, re
from typing import Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from retriever import C

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

SYSTEM_PROMPT = """Jesteś ekspertem od Konstytucji Rzeczypospolitej Polskiej.
Twoim zadaniem jest odpowiadanie na pytania dotyczące Konstytucji RP.

ZASADY:
1. Odpowiadaj WYŁĄCZNIE na podstawie podanego kontekstu z artykułów Konstytucji.
2. ZAWSZE podawaj numer artykułu (np. "Zgodnie z Art. 127 Konstytucji RP...").
3. Jeśli pytanie dotyczy kilku artykułów, wymień wszystkie.
4. Jeśli nie znajdziesz odpowiedzi w podanym kontekście — odpowiedz dokładnie:
   "Nie znalazłem odpowiedzi w Konstytucji RP na podstawie dostępnych artykułów."
5. Nie dodawaj informacji spoza kontekstu.
6. Odpowiadaj po polsku, zwięźle i precyzyjnie.
7. Strukturyzuj odpowiedź: najpierw bezpośrednia odpowiedź, potem szczegóły."""


def build_context(chunks: list) -> str:
    lines = []
    for c in chunks:
        lines.append(f"[{c['chapter']}]")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines)


def build_prompt(question: str, chunks: list, history: list = None) -> str:
    context = build_context(chunks)
    history_text = ""
    if history:
        for turn in history[-3:]:
            history_text += f"Użytkownik: {turn['q']}\nAsystent: {turn['a']}\n\n"

    prompt = f"""KONTEKST Z KONSTYTUCJI RP:
{context}

{f"HISTORIA ROZMOWY:{chr(10)}{history_text}" if history_text else ""}
PYTANIE: {question}

ODPOWIEDŹ (oparta wyłącznie na powyższym kontekście):"""
    return prompt


def query_ollama(question: str, chunks: list, history: list = None, stream: bool = False) -> str:
    if not REQUESTS_AVAILABLE:
        return fallback_answer(question, chunks)

    prompt = build_prompt(question, chunks, history)

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_ctx": 4096,
                }
            },
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return fallback_answer(question, chunks)
    except Exception as e:
        return C.red(f"[Błąd Ollama: {e}]") + "\n\n" + fallback_answer(question, chunks)


def fallback_answer(question: str, chunks: list, colored: bool = True) -> str:
    if not chunks:
        msg = "Nie znalazłem odpowiedzi w Konstytucji RP."
        return C.red(msg) if colored else msg

    lines = [
        C.bold("Odpowiedź na podstawie Konstytucji RP") if colored
        else "Odpowiedź na podstawie Konstytucji RP",
        (C.yellow("Tryb: bezpośrednie cytowanie – Ollama niedostępna") if colored
         else "Tryb: bezpośrednie cytowanie – Ollama niedostępna") + "\n",
    ]

    for c in chunks:
        score = c.get("score", 0.0)
        if colored:
            conf_str = C.green("Wysoka") if score >= 0.20 else C.yellow("Średnia") if score >= 0.10 else C.red("Niska")
        else:
            conf_str = "Wysoka" if score >= 0.20 else "Średnia" if score >= 0.10 else "Niska"

        lines.append("-" * 60)
        lines.append((C.bold(c["chapter"]) if colored else c["chapter"]))
        lines.append(
            (C.bold(f"Art. {c['art_num']}") if colored else f"Art. {c['art_num']}") +
            f"  trafność: {score:.1%}  pewność: {conf_str}"
        )
        lines.append(c["text"] + "\n")

    return "\n".join(lines)


def check_ollama_status() -> dict:
    if not REQUESTS_AVAILABLE:
        return {"available": False, "reason": "brak biblioteki requests"}
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        return {"available": True, "models": models, "url": OLLAMA_URL}
    except Exception as e:
        return {"available": False, "reason": str(e)}