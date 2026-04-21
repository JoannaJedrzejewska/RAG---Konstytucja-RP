# RAG---Konstytucja-RP
System RAG (*Retrieval-Augmented Generation*) do zadawania pytań o treść  
**Konstytucji Rzeczypospolitej Polskiej z dnia 2 kwietnia 1997 r.**

---

## Struktura projektu

```
konstytucja_rag/
├── indexer.py          # parsowanie PDF → chunki → indeks TF-IDF
├── retriever.py        # wyszukiwanie semantyczne (TF-IDF + n-gramy)
├── llm.py              # integracja z Ollama (+ fallback bez LLM)
├── cli.py              # interfejs konsolowy (REPL)
├── app.py              # interfejs webowy (Gradio)
├── requirements.txt    # zależności Pythona
├── chunks.json         # 243 artykuły (auto-generowany przez indexer)
├── tfidf_vectorizer.pkl # model TF-IDF (auto-generowany)
├── tfidf_matrix.pkl    # macierz wektorów (auto-generowany)
└── history.jsonl       # historia zapytań (auto-generowany przez cli)
```

---

### Instalacja zależności

```bash
pip install -r requirements.txt
```
### Budowanie indeksu (jednorazowo)
```bash
python indexer.py D19970483Lj.pdf
```

### Interfejs konsolowy (CLI)

```bash
python cli.py                               # włączenie
python cli.py -q "Kto powołuje premiera?"   # jednorazowe pytanie
python cli.py --top-k 7                     # więcej wyników
python cli.py --no-llm                      # bez Ollama (samo cytowanie)
python cli.py --model gemma2:9b             # inny model Ollama
```

### Interfejs webowy (Gradio)

```bash
pip install gradio
python app.py
# Otwórz przeglądarkę: http://localhost:7860
```

---

```bash
python app.py
# Otwórz przeglądarkę: http://localhost:7860
```

---

##  Konfiguracja Ollama 

> **Bez Ollamy** aplikacja działa w trybie fallback — cytuje artykuły
> bezpośrednio bez interpretacji LLM. Wystarczy odznaczyć
> „Użyj Ollama LLM" w prawym panelu aplikacji.

### Instalacja

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# macOS (Homebrew)
brew install ollama

# Windows
# Pobierz instalator .exe: https://ollama.com/download
```

### Wybór modelu według dostępnego RAM

Przed pobraniem modelu sprawdź ile masz wolnej pamięci, jako początek rekomenduję najleszy model gemma2:2b
Zainstalujesz go następująco:

```bash
free -h    # sprawdź pamięć
ollama pull gemma2:2b
```

### Uruchomienie serwera

```bash
# jeśli nie startuje automatycznie:
ollama serve

```

### Weryfikacja

```bash
ollama list                          # lista pobranych modeli
curl http://localhost:11434/api/tags # sprawdź czy API działa
```

W aplikacji kliknij **„Odśwież status Ollamy"** — baner zmieni się na zielony:
Ollama dostępna | Modele: gemma2:2b


Jeśli widzisz błąd `404 Not Found` — model nie jest pobrany. Uruchom `ollama pull <nazwa>`.

### Zmienne środowiskowe (opcjonalne)

```bash
export OLLAMA_URL=http://localhost:11434   # domyślny adres
export OLLAMA_MODEL=gemma2:2b             # model do użycia
```

Upewnij się że nazwa w `OLLAMA_MODEL` **dokładnie odpowiada** nazwie widocznej w `ollama list`.

### UWAGA

Modele LLM działające lokalnie na CPU są **znacznie wolniejsze** niż w chmurze.
Czas odpowiedzi zależy od sprzętu:


---

## Przykładowe pytania

| Pytanie | Artykuł |
|---------|---------|
| Kto może być prezydentem Polski? | Art. 127 |
| Kto powołuje premiera? | Art. 154 |
| Ile trwa kadencja Sejmu? | Art. 98 |
| Kiedy można wprowadzić stan wyjątkowy? | Art. 230 |
| Jakie prawa ma obywatel polski? | Art. 30-86 |
| Jak zmienić Konstytucję? | Art. 235 |
| Co to jest Trybunał Konstytucyjny? | Art. 188-197 |
| Kto sprawuje władzę ustawodawczą? | Art. 95-96 |

---

## Komendy CLI

```
/help          – wyświetl pomoc
/exit          – zakończ program
/historia      – pokaż ostatnie zapytania
/reset         – wyczyść historię konwersacji
/rozdzial XI   – filtruj tylko po rozdziale XI (Stany nadzwyczajne)
/top 7         – zmień liczbę wyników na 7
/status        – sprawdź połączenie z Ollama
```