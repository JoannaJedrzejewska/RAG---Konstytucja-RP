import argparse, sys, os, json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from retriever import retrieve_smart
from llm import query_ollama, fallback_answer, check_ollama_status

HISTORY_FILE = os.path.join(BASE_DIR, "history.jsonl")

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          KONSTYTUCJA RP – System Pytań i Odpowiedzi          ║
║      Źródło: Konstytucja RP z dnia 2 kwietnia 1997 r.        ║
╚══════════════════════════════════════════════════════════════╝
Komendy: /exit  /historia  /reset  /rozdzial <numer>  /top <k>
         /status  /help
"""

HELP_TEXT = """
DOSTĘPNE KOMENDY:
  /exit               – zakończ program
  /historia           – pokaż historię rozmowy
  /reset              – wyczyść historię
  /rozdzial <numer>   – filtruj wg rozdziału (np. /rozdzial XI)
  /top <liczba>       – zmień liczbę wyników (np. /top 7)
  /status             – status połączenia z Ollama
  /help               – ta pomoc

PRZYKŁADOWE PYTANIA:
  Kto może być prezydentem Polski?
  Kto powołuje premiera?
  Ile trwa kadencja Sejmu?
  Kiedy można wprowadzić stan wyjątkowy?
  Jakie prawa ma obywatel polski?
  Jak zmienić Konstytucję?
  Co to jest Trybunał Stanu?
"""

def print_sources(chunks):
    print("\nŹródła:")
    for c in chunks:
        print(f"   • Art. {c['art_num']:>4} | {c['chapter']:<55} | {c['confidence']} ({c['score']:.1%})")

def save_history(q, a, chunks):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": q,
        "answer": a,
        "sources": [{"art": c["art_num"], "chapter": c["chapter"]} for c in chunks]
    }
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    history = []
    with open(HISTORY_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                history.append(json.loads(line))
            except:
                pass
    return history

def ask(question, top_k, chapter_filter, use_llm, conversation_history):
    chunks, detected_chapter = retrieve_smart(question, top_k=top_k, highlight=False)

    if not chunks:
        return "Nie znalazłem odpowiedzi w Konstytucji RP na podstawie dostępnych artykułów.", []

    if use_llm:
        answer = query_ollama(question, chunks, history=conversation_history)
    else:
        answer = fallback_answer(question, chunks)

    return answer, chunks

def run_interactive(args):
    print(BANNER)

    # Status Ollama
    use_llm = not args.no_llm
    status = check_ollama_status()
    if use_llm:
        if status["available"]:
            models = ", ".join(status["models"][:3])
            print(f"Ollama dostępna ({status['url']}) | Modele: {models}")
        else:
            print(f"Ollama niedostępna ({status.get('reason','?')}) – tryb fallback (cytowanie artykułów)")
            use_llm = False
    else:
        print("Tryb --no-llm: bezpośrednie cytowanie artykułów")

    top_k = args.top_k
    chapter_filter = None
    conversation_history = []

    print(f"\nLiczba wyników: {top_k} | Wpisz /help po pomoc\n")

    while True:
        try:
            question = input("Pytanie: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDo widzenia!")
            break

        if not question:
            continue

        # --- Komendy specjalne ---
        if question.lower() in ("/exit", "/quit", "exit", "quit"):
            print("Do widzenia!")
            break

        if question.lower() == "/help":
            print(HELP_TEXT)
            continue

        if question.lower() == "/reset":
            conversation_history.clear()
            print("Historia rozmowy wyczyszczona.")
            continue

        if question.lower() == "/status":
            st = check_ollama_status()
            if st["available"]:
                print(f"Ollama: {st['url']} | Modele: {', '.join(st['models'])}")
            else:
                print(f"Ollama: {st.get('reason','niedostępna')}")
            continue

        if question.lower() == "/historia":
            hist = load_history()
            if not hist:
                print("Brak historii.")
            else:
                for i, h in enumerate(hist[-5:], 1):
                    print(f"\n[{i}] {h['timestamp'][:16]}")
                    print(f"    P: {h['question']}")
                    srcs = ", ".join(f"Art.{s['art']}" for s in h['sources'][:3])
                    print(f"    Źródła: {srcs}")
            continue

        if question.lower().startswith("/rozdzial "):
            chapter_filter = question.split()[-1].upper()
            print(f"Filtr rozdziału ustawiony na: {chapter_filter}")
            continue

        if question.lower().startswith("/top "):
            try:
                top_k = int(question.split()[-1])
                print(f"Liczba wyników: {top_k}")
            except ValueError:
                print("Użyj: /top <liczba>")
            continue

        print("\n ...Szukam w Konstytucji...")
        answer, chunks = ask(question, top_k, chapter_filter, use_llm, conversation_history)

        print("\n" + "═" * 65)
        print("ODPOWIEDŹ:\n")
        print(answer)
        print_sources(chunks)
        print("═" * 65 + "\n")

        # Zapis historii
        if chunks:
            conversation_history.append({"q": question, "a": answer})
            save_history(question, answer, chunks)

def run_single(args):
    use_llm = not args.no_llm
    if use_llm:
        st = check_ollama_status()
        use_llm = st["available"]

    answer, chunks = ask(args.query, args.top_k, None, use_llm, [])
    print(answer)
    print("\nŹródła:")
    for c in chunks:
        print(f"   Art. {c['art_num']} | {c['chapter']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Konstytucja RP – Q&A System")
    parser.add_argument("-q", "--query", help="Jednorazowe pytanie")
    parser.add_argument("--top-k", type=int, default=5, help="Liczba wyników (domyślnie: 5)")
    parser.add_argument("--no-llm", action="store_true", help="Wyłącz Ollama, tryb fallback")
    parser.add_argument("--model", default=None, help="Model Ollama (np. llama3.1:8b)")
    args = parser.parse_args()

    if args.model:
        import llm as llm_module
        llm_module.OLLAMA_MODEL = args.model

    if args.query:
        run_single(args)
    else:
        run_interactive(args)