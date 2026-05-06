from flask import Flask, request, jsonify, send_file, render_template_string
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import html

app = Flask(__name__)

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

print("Cargando modelo Qwen2-0.5B-Instruct...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
print("Modelo cargado exitosamente.")


def strip_html(raw_html):
    clean = re.sub(r'<.*?>', '', raw_html)
    return html.unescape(clean)


def parse_sentiment(text):
    sentiment_match = re.search(r'Sentiment:\s*(Positive|Negative|Neutral)', text, re.IGNORECASE)
    reason_match = re.search(r'Reason:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)
    return {
        "sentiment": sentiment_match.group(1).strip() if sentiment_match else "Unknown",
        "reason": reason_match.group(1).strip() if reason_match else text.strip()
    }


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    texto = data.get("texto", "").strip()

    if not texto:
        return jsonify({"sentiment": "", "reason": ""})

    prompt_sistema = (
        "You are a financial sentiment analysis expert analyzing SEC filing text. "
        "Analyze the sentiment and respond EXACTLY in this format:\n"
        "Sentiment: [Positive / Negative / Neutral]\n"
        "Reason: [Brief explanation in Spanish]\n\n"
        "Be concise. Only use the format above."
    )

    messages = [
        {"role": "system", "content": prompt_sistema},
        {"role": "user", "content": f"Analyze the sentiment of this SEC filing section:\n\n{texto}"},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
        )

    respuesta = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    parsed = parse_sentiment(respuesta.strip())
    return jsonify(parsed)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    raw = file.read().decode("utf-8", errors="ignore")

    if file.filename.endswith((".html", ".htm")):
        raw = strip_html(raw)

    return jsonify({"text": raw.strip()})


if __name__ == "__main__":
    print("Servidor iniciado en http://localhost:5000")
    app.run(debug=False, port=5000)
