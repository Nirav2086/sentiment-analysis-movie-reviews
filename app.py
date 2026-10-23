"""Simple web demo with Gradio.

Usage:
    python app.py
"""
import gradio as gr

from sentiment_tool import LABELS, MODEL_PATH, clean_text, load_model

model = load_model(MODEL_PATH)


def analyze(text: str):
    if not text or not text.strip():
        return {}
    probs = model.predict_proba([clean_text(text)])[0]
    return {label: float(p) for label, p in zip(LABELS, probs)}


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(lines=4, placeholder="Type or paste a movie review or tweet..."),
    outputs=gr.Label(num_top_classes=3, label="Sentiment"),
    title="Sentiment Analysis Tool for Movie Reviews",
    description="Classifies text as negative, neutral or positive.",
    examples=[
        ["An absolute masterpiece. The acting and the score were breathtaking!"],
        ["Boring, predictable, and way too long."],
        ["Not bad, but not great either."],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
