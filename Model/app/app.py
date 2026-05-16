"""Lightweight Gradio demo for the final HouseMD classifiers.

This app intentionally uses only the selected joblib models included in the
clean repo export. It does not require Torch or Transformers.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
TASKS = ["intent", "emotion", "diagnosis_stage"]

LINEAR_DIR = RUNS / "v2_linear_model_search" / "score"
TFIDF_DIR = RUNS / "v2_tfidf_task_ensemble" / "score"


def build_runtime_features(text: str, speaker: str = "unknown", previous: str = "") -> dict[str, str]:
    text = text.strip()
    speaker = speaker.strip() or "unknown"
    previous = previous.strip()

    feature_parts = [f"[SPEAKER={speaker}]"]
    if previous:
        feature_parts.append(f"[PREV={previous}]")
    feature_parts.append(f"[TEXT={text}]")

    return {
        "text": text,
        "text_only": text,
        "speaker_text": f"[SPEAKER={speaker}] [TEXT={text}]",
        "context_text": f"[PREV={previous}] [TEXT={text}]" if previous else f"[TEXT={text}]",
        "entity_text": f"[SYMPTOM=] [TEST=] [DRUG=] [PROCEDURE=] [ORGAN=] [TEXT={text}]",
        "compact_feature_text": (
            f"[SPEAKER={speaker}] [SYMPTOM=] [TEST=] [DRUG=] [PROCEDURE=] [ORGAN=] [TEXT={text}]"
        ),
        "feature_text": " ".join(feature_parts),
    }


def load_models() -> dict[str, dict]:
    paths = {
        "intent": LINEAR_DIR / "selected_intent.joblib",
        "emotion": TFIDF_DIR / "model_feature_text_soft_vote_emotion.joblib",
        "diagnosis_stage": LINEAR_DIR / "selected_diagnosis_stage.joblib",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model file(s):\n" + "\n".join(missing))
    return {task: joblib.load(path) for task, path in paths.items()}


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(float)
    scores = scores - np.max(scores)
    exp = np.exp(scores)
    denom = exp.sum()
    return exp / denom if denom else np.ones_like(exp) / len(exp)


def predict_bundle(bundle: dict, features: dict[str, str], top_k: int = 3) -> dict[str, float]:
    text_col = bundle.get("text_col", "feature_text")
    x = [features.get(text_col, features["feature_text"])]
    labels = bundle["labels"]

    if "word" in bundle and "char" in bundle:
        probs = (bundle["word"].predict_proba(x)[0] + bundle["char"].predict_proba(x)[0]) / 2.0
        class_labels = list(bundle["word"].named_steps["clf"].classes_)
        scores = np.zeros(len(labels), dtype=float)
        for idx, label in enumerate(labels):
            if label in class_labels:
                scores[idx] = probs[class_labels.index(label)]
    else:
        model = bundle["model"]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]
            class_labels = list(model.classes_)
            scores = np.zeros(len(labels), dtype=float)
            for idx, label in enumerate(labels):
                if label in class_labels:
                    scores[idx] = probs[class_labels.index(label)]
        elif hasattr(model, "decision_function"):
            raw = model.decision_function(x)
            raw = raw[0] if raw.ndim > 1 else np.array([-raw[0], raw[0]])
            class_labels = list(model.classes_)
            aligned = np.zeros(len(labels), dtype=float)
            for idx, label in enumerate(labels):
                if label in class_labels:
                    aligned[idx] = raw[class_labels.index(label)]
            scores = _softmax(aligned)
        else:
            pred = model.predict(x)[0]
            scores = np.array([1.0 if label == pred else 0.0 for label in labels], dtype=float)

    idxs = np.argsort(scores)[::-1][:top_k]
    return {labels[int(i)]: float(scores[int(i)]) for i in idxs}


def main() -> int:
    import gradio as gr

    models = load_models()

    def infer(text: str, speaker: str, previous: str):
        features = build_runtime_features(text, speaker, previous)
        return tuple(predict_bundle(models[task], features) for task in TASKS)

    iface = gr.Interface(
        fn=infer,
        inputs=[
            gr.Textbox(lines=3, label="Text", placeholder="Turkce House MD repligi yazin..."),
            gr.Textbox(lines=1, label="Speaker", value="unknown"),
            gr.Textbox(lines=2, label="Previous context", placeholder="Opsiyonel onceki replik"),
        ],
        outputs=[gr.Label(num_top_classes=3, label=task) for task in TASKS],
        title="House MD Turkce NLP",
        description="Final joblib backend. Outputs: intent, emotion, diagnosis_stage.",
        examples=[
            ["Hemen kontrastli bir beyin MR'i isteyin.", "House", ""],
            ["Beyninde bir tenya var, hicbir sey yapmazsak hafta sonunu goremezsin.", "House", ""],
        ],
    )
    iface.launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
