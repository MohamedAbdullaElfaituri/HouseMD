from __future__ import annotations

import json
import math
import re
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_housemd_diagnosis_model.joblib"
HOST = "127.0.0.1"
PORT = 8501

TEXT_FEATURE_COLUMNS = ["text", "Symptom", "Test", "Drug", "Procedure", "Organ"]
META_FEATURE_COLUMNS = ["speaker", "Intent", "diagnosis_stage", "Emotion", "Sarcasm"]
CASE_CONTEXT_TOKEN_LIMIT = 320
TURKISH_CHARS = "çğıöşü"

MODEL_PACKAGE: dict[str, Any] | None = None


def load_model_package() -> dict[str, Any]:
    global MODEL_PACKAGE
    if MODEL_PACKAGE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}")
        MODEL_PACKAGE = joblib.load(MODEL_PATH)
    return MODEL_PACKAGE


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("\u0307", "")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(f"[^0-9a-z{TURKISH_CHARS}\\s\\-/+%.]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_medical_entities(value: Any) -> str:
    if value is None or not str(value).strip():
        return ""

    raw = str(value).strip().replace('""', '"')
    tokens: list[str] = []
    entity_texts = re.findall(r'"text"\s*:\s*"([^"]+)"', raw)
    entity_types = re.findall(r'"type"\s*:\s*"([^"]+)"', raw)

    tokens.extend(normalize_text(text) for text in entity_texts)
    tokens.extend("entity_" + normalize_text(kind).replace(" ", "_") for kind in entity_types)
    return " ".join(token for token in tokens if token)


def unique_token_join(values: list[str], limit: int = CASE_CONTEXT_TOKEN_LIMIT) -> str:
    seen: list[str] = []
    used = set()
    for value in values:
        for token in str(value).split():
            if token and token not in used:
                seen.append(token)
                used.add(token)
            if len(seen) >= limit:
                return " ".join(seen)
    return " ".join(seen)


def build_row_text(fields: dict[str, Any]) -> str:
    parts: list[str] = []

    for column in TEXT_FEATURE_COLUMNS:
        value = normalize_text(fields.get(column, ""))
        if value:
            parts.append(value)

    for column in META_FEATURE_COLUMNS:
        value = normalize_text(fields.get(column, ""))
        if value:
            parts.append(f"{column.lower()}_{value.replace(' ', '_')}")

    entities = extract_medical_entities(fields.get("medical_entities", ""))
    if entities:
        parts.append(entities)

    return " ".join(parts)


def build_case_context(fields: dict[str, Any]) -> str:
    fact_values = [fields.get(column, "") for column in ["Symptom", "Test", "Drug", "Procedure", "Organ"]]
    entity_values = [fields.get("medical_entities", "")]
    fact_text = unique_token_join([normalize_text(value) for value in fact_values])
    entity_text = unique_token_join([extract_medical_entities(value) for value in entity_values])
    return f"{fact_text} {entity_text}".strip()


def softmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = float(np.sum(exp_values))
    if not math.isfinite(total) or total <= 0:
        return np.zeros_like(values, dtype=float)
    return exp_values / total


def predict(fields: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    package = load_model_package()
    pipeline = package["pipeline"]

    row_text = build_row_text(fields)
    case_context = normalize_text(fields.get("case_context", "")) or build_case_context(fields)
    model_text = f"{row_text} vaka_baglam {case_context}".strip()

    if not model_text:
        raise ValueError("En az bir metin veya klinik alan doldurun.")

    classes = np.array(pipeline.classes_)
    score_label = "Olasılık"
    if hasattr(pipeline, "predict_proba"):
        scores = np.asarray(pipeline.predict_proba([model_text])[0], dtype=float)
    elif hasattr(pipeline, "decision_function"):
        raw_scores = np.asarray(pipeline.decision_function([model_text]), dtype=float)
        scores = softmax(raw_scores[0] if raw_scores.ndim == 2 else raw_scores)
        score_label = "Skor"
    else:
        prediction = str(pipeline.predict([model_text])[0])
        scores = np.zeros(len(classes), dtype=float)
        if prediction in classes:
            scores[int(np.where(classes == prediction)[0][0])] = 1.0

    top_k = max(1, min(int(top_k), len(classes)))
    order = np.argsort(scores)[::-1][:top_k]
    predictions = [
        {
            "rank": index + 1,
            "label": str(classes[class_index]),
            "score": round(float(scores[class_index]), 6),
            "percent": round(float(scores[class_index]) * 100, 2),
        }
        for index, class_index in enumerate(order)
    ]

    return {
        "best_model_name": package.get("best_model_name", ""),
        "saved_at": package.get("saved_at", ""),
        "score_label": score_label,
        "input_text": model_text,
        "predictions": predictions,
        "summary": package.get("data_summary", {}),
    }


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def html_response(handler: BaseHTTPRequestHandler, html: str) -> None:
    body = html.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class AppHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            html_response(self, INDEX_HTML)
            return
        if path == "/health":
            try:
                package = load_model_package()
                json_response(self, 200, {"ok": True, "model": package.get("best_model_name", "")})
            except Exception as exc:
                json_response(self, 500, {"ok": False, "error": str(exc)})
            return
        json_response(self, 404, {"ok": False, "error": "Bulunamadı."})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/predict":
            json_response(self, 404, {"ok": False, "error": "Bulunamadı."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            fields = payload.get("fields", {})
            top_k = payload.get("top_k", 5)
            result = predict(fields, top_k=top_k)
            json_response(self, 200, {"ok": True, "result": result})
        except Exception as exc:
            json_response(self, 400, {"ok": False, "error": str(exc)})


INDEX_HTML = r"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>House M.D. NLP Model</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --surface: #ffffff;
      --surface-soft: #eef2f6;
      --text: #17202a;
      --muted: #667085;
      --line: #d9e0e8;
      --primary: #285f74;
      --primary-strong: #1f4a5c;
      --accent: #b05d3b;
      --success: #27735f;
      --shadow: 0 16px 40px rgba(23, 32, 42, 0.08);
      --radius: 8px;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    button,
    input,
    select,
    textarea {
      font: inherit;
    }

    .shell {
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding-bottom: 18px;
      border-bottom: 1px solid var(--line);
    }

    .brand {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    h1 {
      margin: 0;
      font-size: clamp(24px, 3vw, 36px);
      line-height: 1.1;
      font-weight: 740;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }

    .status {
      min-width: 180px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 9px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--surface);
      color: var(--success);
      font-size: 13px;
      font-weight: 650;
      white-space: nowrap;
    }

    .status::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: currentColor;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      gap: 18px;
      margin-top: 20px;
      align-items: start;
    }

    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }

    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 18px 20px;
      border-bottom: 1px solid var(--line);
    }

    .panel-title {
      margin: 0;
      font-size: 16px;
      font-weight: 720;
    }

    .panel-body {
      padding: 20px;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }

    .field {
      display: flex;
      flex-direction: column;
      gap: 7px;
    }

    .field.full {
      grid-column: 1 / -1;
    }

    label {
      color: #344054;
      font-size: 13px;
      font-weight: 650;
    }

    textarea,
    input,
    select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #fbfcfd;
      color: var(--text);
      outline: none;
      padding: 11px 12px;
      min-height: 42px;
      transition: border-color 160ms ease, box-shadow 160ms ease, background 160ms ease;
    }

    textarea {
      resize: vertical;
      min-height: 132px;
      line-height: 1.45;
    }

    textarea:focus,
    input:focus,
    select:focus {
      border-color: var(--primary);
      background: #ffffff;
      box-shadow: 0 0 0 3px rgba(40, 95, 116, 0.14);
    }

    .actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-top: 18px;
      flex-wrap: wrap;
    }

    .button-row {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    button {
      min-height: 42px;
      border: 1px solid transparent;
      border-radius: var(--radius);
      padding: 0 16px;
      cursor: pointer;
      font-weight: 720;
      transition: transform 120ms ease, background 160ms ease, border-color 160ms ease;
    }

    button:active {
      transform: translateY(1px);
    }

    .primary {
      background: var(--primary);
      color: #ffffff;
    }

    .primary:hover {
      background: var(--primary-strong);
    }

    .secondary {
      background: #ffffff;
      color: var(--text);
      border-color: var(--line);
    }

    .secondary:hover {
      border-color: #b9c4d0;
      background: #f8fafc;
    }

    .topk {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }

    .topk select {
      width: 76px;
      min-height: 38px;
      padding: 8px 10px;
      background: #ffffff;
    }

    .result-empty {
      min-height: 280px;
      display: grid;
      place-items: center;
      color: var(--muted);
      text-align: center;
      border: 1px dashed #c9d2dc;
      border-radius: var(--radius);
      background: #fbfcfd;
      padding: 24px;
      line-height: 1.5;
    }

    .prediction-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .prediction {
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #fbfcfd;
    }

    .prediction-top {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }

    .prediction-name {
      min-width: 0;
      font-size: 15px;
      font-weight: 740;
      overflow-wrap: anywhere;
    }

    .prediction-score {
      color: var(--primary);
      font-variant-numeric: tabular-nums;
      font-size: 14px;
      font-weight: 760;
      white-space: nowrap;
    }

    .bar {
      height: 8px;
      overflow: hidden;
      border-radius: 999px;
      background: var(--surface-soft);
    }

    .bar span {
      display: block;
      height: 100%;
      width: var(--width);
      border-radius: inherit;
      background: linear-gradient(90deg, var(--primary), var(--accent));
    }

    .best {
      border-color: rgba(40, 95, 116, 0.32);
      background: #f7fafb;
    }

    .meta {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
    }

    .metric {
      padding: 12px;
      border-radius: var(--radius);
      background: #f5f7fa;
      border: 1px solid #e1e6ed;
    }

    .metric-value {
      display: block;
      font-size: 18px;
      line-height: 1.2;
      font-weight: 760;
      font-variant-numeric: tabular-nums;
    }

    .metric-label {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }

    .error {
      border: 1px solid rgba(176, 93, 59, 0.35);
      background: #fff8f5;
      color: #8f3f25;
      border-radius: var(--radius);
      padding: 12px;
      line-height: 1.45;
      font-size: 14px;
      font-weight: 650;
    }

    .small-note {
      margin: 16px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }

    @media (max-width: 920px) {
      .layout {
        grid-template-columns: 1fr;
      }

      .topbar {
        align-items: flex-start;
        flex-direction: column;
      }

      .status {
        min-width: 0;
      }
    }

    @media (max-width: 640px) {
      .shell {
        width: min(100% - 20px, 1180px);
        padding-top: 16px;
      }

      .grid,
      .meta {
        grid-template-columns: 1fr;
      }

      .panel-header,
      .panel-body {
        padding: 16px;
      }

      .actions {
        align-items: stretch;
        flex-direction: column;
      }

      .button-row,
      .button-row button,
      .topk {
        width: 100%;
      }

      .button-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
      }

      .topk {
        justify-content: space-between;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand">
        <h1>House M.D. NLP Model</h1>
        <p class="subtitle">Vaka metninden tanı etiketi tahmini</p>
      </div>
      <div class="status" id="status">Model hazır</div>
    </header>

    <section class="layout">
      <form class="panel" id="predict-form">
        <div class="panel-header">
          <h2 class="panel-title">Vaka Bilgileri</h2>
        </div>
        <div class="panel-body">
          <div class="grid">
            <div class="field full">
              <label for="text">Vaka metni</label>
              <textarea id="text" name="text" placeholder="Hasta nöbet geçirdi, MR görüntüsünde beyinde lezyon var."></textarea>
            </div>

            <div class="field">
              <label for="Symptom">Semptom</label>
              <input id="Symptom" name="Symptom" placeholder="nöbet">
            </div>
            <div class="field">
              <label for="Test">Test</label>
              <input id="Test" name="Test" placeholder="MR">
            </div>
            <div class="field">
              <label for="Drug">İlaç</label>
              <input id="Drug" name="Drug" placeholder="kortikosteroid">
            </div>
            <div class="field">
              <label for="Procedure">Prosedür</label>
              <input id="Procedure" name="Procedure" placeholder="biyopsi">
            </div>
            <div class="field">
              <label for="Organ">Organ</label>
              <input id="Organ" name="Organ" placeholder="beyin">
            </div>
            <div class="field">
              <label for="speaker">Konuşmacı</label>
              <input id="speaker" name="speaker" placeholder="Wilson">
            </div>
            <div class="field">
              <label for="Intent">Niyet</label>
              <select id="Intent" name="Intent">
                <option value="">Seçiniz</option>
                <option>açıklama</option>
                <option>soru</option>
                <option>öneri</option>
                <option>itiraz</option>
              </select>
            </div>
            <div class="field">
              <label for="diagnosis_stage">Tanı aşaması</label>
              <select id="diagnosis_stage" name="diagnosis_stage">
                <option value="">Seçiniz</option>
                <option>hipotez</option>
                <option>test</option>
                <option>tedavi</option>
                <option>sonuç</option>
              </select>
            </div>
            <div class="field">
              <label for="Emotion">Duygu</label>
              <select id="Emotion" name="Emotion">
                <option value="">Seçiniz</option>
                <option>nötr</option>
                <option>endişeli</option>
                <option>emin</option>
                <option>şüpheli</option>
              </select>
            </div>
            <div class="field">
              <label for="Sarcasm">Sarkazm</label>
              <select id="Sarcasm" name="Sarcasm">
                <option value="">Seçiniz</option>
                <option value="0">0</option>
                <option value="1">1</option>
              </select>
            </div>
          </div>

          <div class="actions">
            <div class="button-row">
              <button class="primary" type="submit">Tahmin Et</button>
              <button class="secondary" type="button" id="clear">Temizle</button>
            </div>
            <label class="topk" for="top_k">Sonuç sayısı
              <select id="top_k">
                <option>3</option>
                <option selected>5</option>
                <option>10</option>
              </select>
            </label>
          </div>
        </div>
      </form>

      <aside class="panel">
        <div class="panel-header">
          <h2 class="panel-title">Tahmin</h2>
        </div>
        <div class="panel-body">
          <div id="result" class="result-empty">Vaka bilgilerini girip tahmin alın.</div>
          <div class="meta" id="metrics" hidden></div>
          <p class="small-note">Bu çıktı eğitim amaçlıdır; klinik tanı veya tedavi önerisi değildir.</p>
        </div>
      </aside>
    </section>
  </main>

  <script>
    const form = document.querySelector("#predict-form");
    const result = document.querySelector("#result");
    const metrics = document.querySelector("#metrics");
    const statusEl = document.querySelector("#status");
    const clearButton = document.querySelector("#clear");

    const fieldNames = [
      "text", "Symptom", "Test", "Drug", "Procedure", "Organ",
      "speaker", "Intent", "diagnosis_stage", "Emotion", "Sarcasm"
    ];

    function readFields() {
      return Object.fromEntries(fieldNames.map((name) => {
        const element = document.querySelector(`[name="${name}"]`);
        return [name, element ? element.value.trim() : ""];
      }));
    }

    function setLoading(isLoading) {
      statusEl.textContent = isLoading ? "Model çalışıyor" : "Model hazır";
      form.querySelectorAll("button, input, select, textarea").forEach((element) => {
        element.disabled = isLoading;
      });
    }

    function showError(message) {
      result.className = "error";
      result.textContent = message;
      metrics.hidden = true;
      metrics.innerHTML = "";
    }

    function metric(label, value) {
      return `
        <div class="metric">
          <span class="metric-value">${value}</span>
          <span class="metric-label">${label}</span>
        </div>
      `;
    }

    function renderResult(payload) {
      const predictions = payload.predictions || [];
      result.className = "prediction-list";
      result.innerHTML = predictions.map((item, index) => {
        const width = Math.max(2, Math.min(100, Number(item.percent || 0)));
        return `
          <div class="prediction ${index === 0 ? "best" : ""}">
            <div class="prediction-top">
              <div class="prediction-name">${item.rank}. ${item.label}</div>
              <div class="prediction-score">${item.percent.toFixed(2)}%</div>
            </div>
            <div class="bar" aria-label="${payload.score_label}: ${item.percent.toFixed(2)}%">
              <span style="--width: ${width}%"></span>
            </div>
          </div>
        `;
      }).join("");

      const summary = payload.summary || {};
      metrics.hidden = false;
      metrics.innerHTML = [
        metric("Model satırı", summary.model_rows || "-"),
        metric("Sınıf", summary.class_count || "-"),
        metric("Test satırı", summary.test_rows || "-"),
        metric("Seçilen model", payload.best_model_name ? "aktif" : "-")
      ].join("");
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      setLoading(true);
      try {
        const response = await fetch("/api/predict", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            fields: readFields(),
            top_k: Number(document.querySelector("#top_k").value)
          })
        });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
          throw new Error(payload.error || "Tahmin alınamadı.");
        }
        renderResult(payload.result);
      } catch (error) {
        showError(error.message);
      } finally {
        setLoading(false);
      }
    });

    clearButton.addEventListener("click", () => {
      form.reset();
      result.className = "result-empty";
      result.textContent = "Vaka bilgilerini girip tahmin alın.";
      metrics.hidden = true;
      metrics.innerHTML = "";
    });
  </script>
</body>
</html>
"""


def run() -> None:
    load_model_package()
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"House M.D. NLP arayüzü: http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
