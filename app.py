"""Flask application providing text summarization and (basic) image context analysis
using Google Gemini models. The frontend (index.html) calls POST /analyze with either
JSON (for summarization) or multipart form-data (for image analysis). This file
implements that API so the webpage shows the summary result correctly.
"""

from flask import Flask, render_template, request, jsonify
import os
import base64
import io
from typing import Optional, Dict, Any

import requests

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # Fallback if library missing

app = Flask(__name__)

# ---------------------- Configuration ----------------------
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
DEFAULT_MODEL = os.getenv("MODEL", "gemini-2.0-flash")  # Preferred
FALLBACK_MODEL = "gemini-1.5-flash"

if not API_KEY:
    print("WARNING: GEMINI_API_KEY not set. Set the env var before using the app.")

_genai_model = None
if genai and API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        _genai_model = genai.GenerativeModel(DEFAULT_MODEL)
        print(f"Initialized genai model: {DEFAULT_MODEL}")
    except Exception as e:  # Model may not exist in library yet
        print(f"genai model init failed ({DEFAULT_MODEL}): {e}")
        try:
            _genai_model = genai.GenerativeModel(FALLBACK_MODEL)
            print(f"Initialized fallback genai model: {FALLBACK_MODEL}")
        except Exception as e2:
            print(f"Fallback model init also failed: {e2}. Will use direct HTTP requests only.")


# ---------------------- Utility Functions ----------------------
def build_summary_prompt(text: str, options: Dict[str, Any]) -> str:
    length = (options.get("length") or "medium").lower()
    tone = (options.get("tone") or "neutral").lower()
    language = options.get("language") or ""

    style_map = {
        "short": "2-3 concise sentences",
        "medium": "a tight single paragraph",
        "detailed": "a detailed yet concise multi-paragraph summary"
    }
    tone_map = {
        "neutral": "neutral, factual prose",
        "bullet": "concise bullet points",
        "technical": "precise technical language"
    }
    style = style_map.get(length, style_map["medium"])
    tone_desc = tone_map.get(tone, tone_map["neutral"])
    lang_part = f" in {language}" if language else ""

    return (
        "You are a careful assistant that preserves facts, figures, and names. "
        f"Summarize the following text as {style} using {tone_desc}{lang_part}. "
        "Highlight critical numbers. If there is uncertainty, mention it briefly.\n\n" + text.strip()
    )


def _genai_summarize(prompt: str) -> Optional[str]:
    if not _genai_model:
        return None
    try:
        resp = _genai_model.generate_content(prompt)
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        # Fallback extraction
        if getattr(resp, "candidates", None):
            cand = resp.candidates[0]
            parts = getattr(cand, "content", {}).parts if getattr(cand, "content", None) else []
            if parts and hasattr(parts[0], "text"):
                return parts[0].text.strip()
    except Exception as e:
        print(f"genai summarization error: {e}")
    return None


def _http_generate(prompt: str, model_name: str) -> Optional[str]:
    if not API_KEY:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, headers={"Content-Type": "application/json", "X-goog-api-key": API_KEY}, json=body, timeout=40)
        if r.status_code == 200:
            data = r.json()
            return data['candidates'][0]['content']['parts'][0]['text'].strip()
        print(f"HTTP model {model_name} error {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"HTTP request failure ({model_name}): {e}")
    return None


def summarize(text: str, options: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_summary_prompt(text, options)
    summary = _genai_summarize(prompt)
    used_model = DEFAULT_MODEL
    source = "genai"
    if not summary:
        summary = _http_generate(prompt, DEFAULT_MODEL)
        source = "http"
    if not summary:
        summary = _http_generate(prompt, FALLBACK_MODEL)
        if summary:
            used_model = FALLBACK_MODEL
            source = "http-fallback"
    if not summary:
        raise RuntimeError("All summarization methods failed")
    return {
        "summary": summary,
        "meta": {"model": used_model, "mode": "summarize", "source": source, "options": options},
        "tokens_used": None
    }


def encode_image(file_storage) -> Dict[str, Any]:
    data = file_storage.read()
    b64 = base64.b64encode(data).decode('utf-8')
    mime = file_storage.mimetype or 'image/png'
    return {"inline_data": {"data": b64, "mime_type": mime}}, data


def image_context(prompt: str, image_part: Dict[str, Any]) -> Optional[str]:
    # Try genai library first for vision
    if _genai_model:
        try:
            resp = _genai_model.generate_content([prompt, image_part])  # type: ignore
            if hasattr(resp, "text") and resp.text:
                return resp.text.strip()
        except Exception as e:
            print(f"genai vision error: {e}")
    # Fallback: treat as text-only description request (no direct image support in HTTP fallback here)
    return None


# ---------------------- Routes ----------------------
@app.route("/")
def index():  # Renders the SPA-like page; dynamic results come from /analyze
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    # JSON summarization request
    if request.content_type and request.content_type.startswith("application/json"):
        data = request.get_json(silent=True) or {}
        mode = data.get("mode") or "summarize"
        if mode != "summarize":
            return jsonify({"error": "Unsupported JSON mode"}), 400
        text = (data.get("text") or "").strip()
        options = data.get("options") or {}
        if not text:
            return jsonify({"error": "No text provided"}), 400
        try:
            result = summarize(text, options)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e), "error_type": type(e).__name__}), 502

    # Multipart image or other form-data
    mode = request.form.get("mode", "")
    if mode == "image_context":
        if "image" not in request.files:
            return jsonify({"error": "Image file missing"}), 400
        image_file = request.files["image"]
        options_raw = request.form.get("options") or "{}"
        try:
            import json
            options = json.loads(options_raw)
        except Exception:
            options = {}
        language = options.get("language") or ""

        image_part, raw_bytes = encode_image(image_file)
        from PIL import Image  # lazy import
        try:
            img = Image.open(io.BytesIO(raw_bytes))
            size_str = f"{img.width}x{img.height}"
        except Exception:
            size_str = "unknown"

        base_prompt = (
            "You are an assistant describing the provided image. "
            "Identify notable objects, text, and context succinctly."
        )
        if language:
            base_prompt += f" Respond in {language}."

        analysis = image_context(base_prompt, image_part)
        if not analysis:
            analysis = "(Vision model unavailable or failed. No analysis produced.)"
        return jsonify({
            "analysis": analysis,
            "tokens_used": None,
            "meta": {
                "model": DEFAULT_MODEL,
                "mode": "image_context",
                "image_size": size_str,
                "options": options
            }
        }), 200

    return jsonify({"error": "Unsupported request format or mode"}), 400


# ---------------------- Entry Point ----------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting server on http://127.0.0.1:{port} (model preference: {DEFAULT_MODEL})")
    app.run(host="0.0.0.0", port=port, debug=True)
