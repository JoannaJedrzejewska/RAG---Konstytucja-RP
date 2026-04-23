# -*- coding: utf-8 -*-

import sys, re, json, pickle, os
import numpy as np

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHAPTERS = {
    "I":    "Rozdział I – RZECZPOSPOLITA",
    "II":   "Rozdział II – WOLNOŚCI, PRAWA I OBOWIĄZKI CZŁOWIEKA I OBYWATELA",
    "III":  "Rozdział III – ŹRÓDŁA PRAWA",
    "IV":   "Rozdział IV – SEJM I SENAT",
    "V":    "Rozdział V – PREZYDENT RZECZYPOSPOLITEJ POLSKIEJ",
    "VI":   "Rozdział VI – RADA MINISTRÓW I ADMINISTRACJA RZĄDOWA",
    "VII":  "Rozdział VII – SAMORZĄD TERYTORIALNY",
    "VIII": "Rozdział VIII – SĄDY I TRYBUNAŁY",
    "IX":   "Rozdział IX – ORGANY KONTROLI PAŃSTWOWEJ I OCHRONY PRAWA",
    "X":    "Rozdział X – FINANSE PUBLICZNE",
    "XI":   "Rozdział XI – STANY NADZWYCZAJNE",
    "XII":  "Rozdział XII – ZMIANA KONSTYTUCJI",
    "XIII": "Rozdział XIII – PRZEPISY PRZEJŚCIOWE I KOŃCOWE",
}


def extract_text_from_pdf(pdf_path: str) -> str:
    if not PYMUPDF_AVAILABLE:
        raise ImportError("Zainstaluj PyMuPDF: pip install pymupdf")
    doc = fitz.open(pdf_path)
    pages = []
    for p in doc:
        pages.append(p.get_text())
    return "".join(pages)


def clean_text(text: str) -> str:
    text = re.sub(r"©Kancelaria Sejmu\s+s\.\s*\d+/\d+\s+\d{4}-\d{2}-\d{2}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def parse_into_chunks(text: str) -> list[dict]:
    chapter_pattern = re.compile(
        r"Rozdział\s+(I{1,3}V?|VI{0,3}|XI{0,3}I?|XII|XIII)\b",
        re.MULTILINE
    )
    chapter_positions = {}
    for m in chapter_pattern.finditer(text):
        roman = m.group(1).strip()
        if roman in CHAPTERS:
            chapter_positions[m.start()] = roman

    art_pattern = re.compile(r"Art\.\s*(\d+[a-z]?)\.", re.MULTILINE)
    art_matches = list(art_pattern.finditer(text))

    if not art_matches:
        print("UWAGA: nie znaleziono żadnych artykułów — sprawdź jakość PDF")
        return []

    chunks = []
    sorted_chapter_pos = sorted(chapter_positions.keys())

    def get_chapter_at(pos: int) -> str:
        """Zwraca nazwę rozdziału obowiązującego w danej pozycji tekstu."""
        current = "PREAMBUŁA"
        for cp in sorted_chapter_pos:
            if cp <= pos:
                current = chapter_positions[cp]
            else:
                break
        return CHAPTERS.get(current, f"Rozdział {current}")

    for i, match in enumerate(art_matches):
        art_num = match.group(1)
        start   = match.start()

        end = art_matches[i + 1].start() if i + 1 < len(art_matches) else len(text)

        art_text = text[start:end].strip()
        art_text = re.sub(r"\n+", " ", art_text)
        art_text = re.sub(r" {2,}", " ", art_text)

        chapter_label = get_chapter_at(start)

        chunks.append({
            "art_num": art_num,
            "chapter": chapter_label,
            "text":    art_text,
        })

    return chunks


def build_tfidf_index(chunks: list[dict]):
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),      # mniej szumu z trigramów
    max_features=10_000,
    sublinear_tf=True,
    min_df=2,                # ignoruj terminy w tylko 1 artykule
    max_df=0.85,             # ignoruj zbyt pospolite słowa
)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def save_artifacts(chunks, vectorizer, matrix):
    chunks_path     = os.path.join(BASE_DIR, "chunks.json")
    vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
    matrix_path     = os.path.join(BASE_DIR, "tfidf_matrix.pkl")

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(matrix_path, "wb") as f:
        pickle.dump(matrix, f)

    print(f"  chunks.json - {chunks_path}")
    print(f"  tfidf_vectorizer.pkl - {vectorizer_path}")
    print(f"  tfidf_matrix.pkl - {matrix_path}")


def main(pdf_path: str):
    print(f"\nWczytywanie PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        print(f"BŁĄD: plik nie istnieje: {pdf_path}")
        sys.exit(1)

    raw   = extract_text_from_pdf(pdf_path)
    text  = clean_text(raw)

    txt_path = os.path.join(BASE_DIR, "konstytucja_clean.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  czysty tekst        → {txt_path}")

    print("✂️  Parsowanie artykułów...")
    chunks = parse_into_chunks(text)
    print(f"  → znaleziono {len(chunks)} artykułów")

    if not chunks:
        print("BŁĄD: brak artykułów — indeks nie został zapisany")
        sys.exit(1)

    for c in chunks[:3]:
        preview = c["text"][:80].replace("\n", " ")
        print(f"    Art. {c['art_num']:>4} | {c['chapter'][:40]} | {preview}...")

    print("Budowanie indeksu TF-IDF...")
    vectorizer, matrix = build_tfidf_index(chunks)
    print(f"\nmacierz: {matrix.shape[0]} dokumentów × {matrix.shape[1]} cech")

    print("Zapisywanie artefaktów...")
    save_artifacts(chunks, vectorizer, matrix)

    print(f"\nGotowe! Zindeksowano {len(chunks)} artykułów.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python indexer.py <ścieżka_do_pdf>")
        print("Przykład: python indexer.py D19970483Lj.pdf")
        sys.exit(1)
    main(sys.argv[1])