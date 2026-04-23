# app.py
# -*- coding: utf-8 -*-

import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import gradio as gr
from retriever import retrieve_smart, CHUNKS
from llm import query_ollama, fallback_answer, check_ollama_status

_conversation_history = []


def _badge(text: str, color: str, bg: str) -> str:
    return (
        f'<span style="'
        f'color:{color};background:{bg};font-weight:700;'
        f'padding:2px 8px;border-radius:4px;font-size:0.82em;'
        f'letter-spacing:0.03em">{text}</span>'
    )

def conf_badge(score: float) -> str:
    if score >= 0.20: return _badge("WYSOKA",  "#166534", "#dcfce7")
    if score >= 0.10: return _badge("ŚREDNIA", "#854d0e", "#fef9c3")
    return              _badge("NISKA",   "#991b1b", "#fee2e2")

def get_ollama_status_html() -> str:
    st = check_ollama_status()
    if st["available"]:
        models = ", ".join(st["models"][:4]) or "brak modeli"
        return (
            f'<div style="padding:8px 12px;background:#f0fdf4;border:1px solid #bbf7d0;'
            f'border-radius:6px;font-size:0.9em">'
            f'<span style="color:#166534;font-weight:700">Ollama dostępna</span>'
            f'<span style="color:#374151"> &nbsp;|&nbsp; Modele: <b>{models}</b></span>'
            f'</div>'
        )
    reason = st.get("reason", "?")
    return (
        f'<div style="padding:8px 12px;background:#fef2f2;border:1px solid #fecaca;'
        f'border-radius:6px;font-size:0.9em">'
        f'<span style="color:#991b1b;font-weight:700">Ollama niedostępna</span>'
        f'<span style="color:#374151"> &nbsp;|&nbsp; {reason}</span><br>'
        f'<span style="color:#6b7280;font-size:0.88em">Tryb: bezpośrednie cytowanie artykułów</span>'
        f'</div>'
    )


def format_sources_html(chunks: list) -> str:
    if not chunks:
        return ""
    rows = ""
    for c in chunks:
        score_pct = f"{c['score']:.1%}"
        chapter_short = c["chapter"].replace("Rozdział ", "Rozdz. ")
        rows += (
            f'<tr style="border-bottom:1px solid #f3f4f6">'
            f'<td style="padding:7px 12px;font-weight:700;white-space:nowrap;'
            f'color:#1e3a5f;text-decoration:underline">Art.&nbsp;{c["art_num"]}</td>'
            f'<td style="padding:7px 12px;font-size:0.85em;color:#4b5563">{chapter_short}</td>'
            f'<td style="padding:7px 12px;text-align:center">{conf_badge(c["score"])}</td>'
            f'<td style="padding:7px 12px;text-align:center;font-size:0.85em;'
            f'font-weight:600;color:#374151">{score_pct}</td>'
            f'</tr>'
        )
    return (
        f'<div style="margin-top:14px">'
        f'<div style="font-weight:700;font-size:0.9em;color:#1e3a5f;'
        f'margin-bottom:6px;text-transform:uppercase;letter-spacing:0.05em">'
        f'Użyte źródła</div>'
        f'<table style="width:100%;border-collapse:collapse;font-size:0.88em;'
        f'border:1px solid #e5e7eb;border-radius:8px;overflow:hidden">'
        f'<thead><tr style="background:#f1f5f9">'
        f'<th style="padding:7px 12px;text-align:left;font-weight:700;color:#1e3a5f">Artykuł</th>'
        f'<th style="padding:7px 12px;text-align:left;font-weight:700;color:#1e3a5f">Rozdział</th>'
        f'<th style="padding:7px 12px;text-align:center;font-weight:700;color:#1e3a5f">Pewność</th>'
        f'<th style="padding:7px 12px;text-align:center;font-weight:700;color:#1e3a5f">Trafność</th>'
        f'</tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'</table></div>'
    )


def format_chunks_detail(chunks: list) -> str:
    if not chunks:
        return ""
    parts = []
    for c in chunks:
        conf_text = (
            "**Wysoka**"  if c["score"] >= 0.20 else
            "**Średnia**" if c["score"] >= 0.10 else
            "**Niska**"
        )
        parts.append(
            f"### Art. {c['art_num']} — {c['chapter']}\n\n"
            f"{c['text']}\n\n"
            f"*Trafność: {c['score']:.1%} | Pewność: {conf_text}*\n\n---\n"
        )
    return "\n".join(parts)


def answer_question(question, top_k, use_ollama, show_chunks, chapter_filter):
    global _conversation_history

    if not question.strip():
        return "", "", ""

    cf = chapter_filter.strip().upper() if chapter_filter.strip() else None
    chunks, detected = retrieve_smart(
        question, top_k=int(top_k), highlight=False, colored=False
    )

    if cf:
        from retriever import retrieve
        chunks_filtered = retrieve(
            question, top_k=int(top_k), chapter_filter=cf, colored=False
        )
        if chunks_filtered:
            chunks = chunks_filtered

    if not chunks:
        no_ans = "Nie znalazłem odpowiedzi w Konstytucji RP na podstawie dostępnych artykułów."
        return no_ans, "", ""

    if use_ollama:
        st = check_ollama_status()
        answer = (
            query_ollama(question, chunks, history=_conversation_history)
            if st["available"]
            else fallback_answer(question, chunks, colored=False)
        )
    else:
        answer = fallback_answer(question, chunks, colored=False)

    _conversation_history.append({"q": question, "a": answer})
    if len(_conversation_history) > 10:
        _conversation_history = _conversation_history[-10:]

    return answer, format_sources_html(chunks), format_chunks_detail(chunks) if show_chunks else ""


def reset_history():
    global _conversation_history
    _conversation_history.clear()
    return gr.update(value="Historia rozmowy wyczyszczona.", visible=True)


EXAMPLE_QUESTIONS = [
    "Kto może być prezydentem Polski?",
    "Kto powołuje premiera?",
    "Ile trwa kadencja Sejmu?",
    "Kiedy można wprowadzić stan wyjątkowy?",
    "Jakie prawa ma obywatel polski?",
    "Jak zmienić Konstytucję?",
    "Co to jest Trybunał Konstytucyjny?",
    "Kto sprawuje władzę ustawodawczą?",
    "Jakie są prawa dziecka?",
    "Czym zajmuje się Rzecznik Praw Obywatelskich?",
]

CHAPTERS_FILTER_CHOICES = [
    "", "I", "II", "III", "IV", "V", "VI", "VII",
    "VIII", "IX", "X", "XI", "XII", "XIII"
]

CSS = """
.answer-box textarea {
    font-size: 1.05em !important;
    line-height: 1.75 !important;
    font-family: 'Inter', sans-serif !important;
}
.submit-btn { background: #1d4ed8 !important; font-weight: 700 !important; }
.danger-btn { background: #dc2626 !important; color: white !important; }
footer { display: none !important; }
"""


def build_ui():
    with gr.Blocks(title="Konstytucja RP – Q&A") as demo:

        gr.HTML("""
        <div style="text-align:center;padding:24px 0 8px">
          <h1 style="font-size:2em;font-weight:800;color:#1e3a5f;
                     letter-spacing:-0.01em;margin-bottom:6px">
            Konstytucja RP &mdash; System Q&amp;A
          </h1>
          <p style="color:#6b7280;font-size:1em">
            Pytaj o Konstytucję Rzeczypospolitej Polskiej z dnia 2 kwietnia 1997 r.
          </p>
          <hr style="margin:14px auto;width:60%;border-color:#e5e7eb">
        </div>
        """)

        ollama_status = gr.HTML(get_ollama_status_html())

        gr.HTML('<div style="margin-top:12px"></div>')

        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="Twoje pytanie",
                    placeholder="np. Kto może być prezydentem Polski?",
                    lines=2,
                    autofocus=True
                )
                with gr.Row():
                    submit_btn = gr.Button(
                        "Szukaj w Konstytucji", variant="primary",
                        scale=3, elem_classes=["submit-btn"]
                    )
                    clear_btn = gr.Button("Wyczyść", scale=1)

                gr.Examples(
                    examples=EXAMPLE_QUESTIONS,
                    inputs=question_input,
                    label="Przykładowe pytania – kliknij, aby wypełnić"
                )

            with gr.Column(scale=1):
                gr.HTML('<div style="font-weight:700;color:#1e3a5f;'
                        'font-size:0.9em;margin-bottom:4px">Ustawienia</div>')
                top_k_slider = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Liczba wyników (top-k)"
                )
                use_ollama_cb  = gr.Checkbox(label="Użyj Ollama LLM", value=True)
                show_chunks_cb = gr.Checkbox(label="Pokaż pełne artykuły", value=False)
                chapter_dd = gr.Dropdown(
                    choices=CHAPTERS_FILTER_CHOICES,
                    value="",
                    label="Filtr rozdziału (opcjonalny)",
                    allow_custom_value=True
                )
                gr.HTML('<hr style="border-color:#e5e7eb;margin:8px 0">')
                refresh_btn = gr.Button("Odśwież status Ollamy")
                reset_btn   = gr.Button(
                    "Resetuj historię rozmowy",
                    elem_classes=["danger-btn"]
                )
                # ── ZMIANA 2: reset_msg zadeklarowany normalnie, domyślnie ukryty ──
                reset_msg = gr.Textbox(
                    label="",
                    interactive=False,
                    visible=False,
                    max_lines=1,
                    container=False,
                )

        gr.HTML('<hr style="border-color:#e5e7eb;margin:16px 0 8px">')

        answer_output = gr.Textbox(
            label="Odpowiedź",
            lines=10,
            interactive=False,
            elem_classes=["answer-box"]
        )
        sources_output = gr.HTML()
        chunks_output  = gr.Markdown(
            label="Pełne artykuły źródłowe",
            visible=True
        )

        inputs  = [question_input, top_k_slider, use_ollama_cb, show_chunks_cb, chapter_dd]
        outputs = [answer_output, sources_output, chunks_output]

        submit_btn.click(fn=answer_question, inputs=inputs, outputs=outputs)
        question_input.submit(fn=answer_question, inputs=inputs, outputs=outputs)
        clear_btn.click(
            fn=lambda: ("", "", "", ""),
            outputs=[question_input, answer_output, sources_output, chunks_output]
        )
        refresh_btn.click(fn=get_ollama_status_html, outputs=ollama_status)
        # ── ZMIANA 3: outputs=reset_msg zamiast inline gr.Textbox() ──
        reset_btn.click(fn=reset_history, outputs=reset_msg)

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Default(
            primary_hue="blue",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
        ),
        css=CSS,
    )