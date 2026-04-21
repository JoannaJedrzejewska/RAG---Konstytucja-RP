import json, pickle, re, os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


#Kolory ANSI
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[1;32m"   #pogrubiona zieleń
    YELLOW = "\033[1;33m"   #pogrubiony żółty
    RED    = "\033[1;31m"   #pogrubiona czerwień

    @staticmethod
    def green(text):  return f"{C.GREEN}{text}{C.RESET}"
    @staticmethod
    def yellow(text): return f"{C.YELLOW}{text}{C.RESET}"
    @staticmethod
    def red(text):    return f"{C.RED}{text}{C.RESET}"
    @staticmethod
    def bold(text):   return f"{C.BOLD}{text}{C.RESET}"


def _load():
    with open(os.path.join(BASE_DIR, "chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    with open(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(BASE_DIR, "tfidf_matrix.pkl"), "rb") as f:
        matrix = pickle.load(f)
    return chunks, vectorizer, matrix


CHUNKS, VECTORIZER, MATRIX = _load()


CHAPTER_ALIASES = {
    "prawa": "II", "wolności": "II", "obowiązki": "II",
    "sejm": "IV", "senat": "IV", "parlament": "IV",
    "prezydent": "V",
    "rząd": "VI", "minister": "VI", "premier": "VI", "rada ministrów": "VI",
    "samorząd": "VII",
    "sąd": "VIII", "trybunał": "VIII",
    "kontrola": "IX", "rzecznik": "IX",
    "finanse": "X", "budżet": "X",
    "stan nadzwyczajny": "XI", "stan wyjątkowy": "XI", "stan wojenny": "XI",
    "zmiana konstytucji": "XII",
}


def detect_chapter(query: str) -> str | None:
    q = query.lower()
    for kw, roman in CHAPTER_ALIASES.items():
        if kw in q:
            return roman
    return None


def highlight_keywords(text: str, query: str) -> str:
    words = [w for w in re.findall(r"\w{4,}", query.lower()) if len(w) > 3]
    for w in words:
        text = re.sub(f"({re.escape(w)})", r"**\1**", text, flags=re.IGNORECASE)
    return text


def confidence_label(score: float) -> str:
    """Zwraca etykietę z kolorem ANSI."""
    if score >= 0.20: return C.green("Wysoka")
    if score >= 0.10: return C.yellow("Średnia")
    return C.red("Niska")


def confidence_label_plain(score: float) -> str:
    """Wersja bez kolorów (do logów, Gradio, JSON)."""
    if score >= 0.20: return "Wysoka"
    if score >= 0.10: return "Średnia"
    return "Niska"


def retrieve(query: str, top_k: int = 5, chapter_filter: str | None = None,
             highlight: bool = False, colored: bool = True):
    q_vec = VECTORIZER.transform([query])
    scores = cosine_similarity(q_vec, MATRIX).flatten()

    if chapter_filter:
        for i, chunk in enumerate(CHUNKS):
            if f"Rozdział {chapter_filter}" not in chunk["chapter"]:
                scores[i] = 0.0

    top_idx = np.argsort(scores)[::-1][:top_k * 2]
    results = []
    seen_arts = set()
    for idx in top_idx:
        if len(results) >= top_k:
            break
        art = CHUNKS[idx]["art_num"]
        if art in seen_arts:
            continue
        seen_arts.add(art)
        if scores[idx] < 0.02:
            continue
        chunk = dict(CHUNKS[idx])
        chunk["score"] = float(scores[idx])
        # colored=True → CLI, colored=False → Gradio / logi
        chunk["confidence"] = (
            confidence_label(chunk["score"]) if colored
            else confidence_label_plain(chunk["score"])
        )
        if highlight:
            chunk["text_highlighted"] = highlight_keywords(chunk["text"], query)
        results.append(chunk)
    return results


def retrieve_smart(query: str, top_k: int = 5, highlight: bool = False,
                   colored: bool = True):
    chapter = detect_chapter(query)
    results = retrieve(query, top_k=top_k, chapter_filter=chapter,
                       highlight=highlight, colored=colored)
    if len(results) < 2:
        results = retrieve(query, top_k=top_k, highlight=highlight, colored=colored)
    return results, chapter