"""
ComfyUI custom nodes for Nano Banana Pro and Nano Banana 2 APIs.
- Nano Banana Pro: generate; Nano Banana Pro Edit: edit with images (gateway.bananapro.site).
- Nano Banana 2: generate; Nano Banana 2 Edit: edit with images (Gemini 2.0 Flash).
Each node returns image as ComfyUI IMAGE tensor [B, H, W, C].
"""

import logging
import io
import time
import base64

import requests
import torch
import numpy as np
from PIL import Image


# ---------- Nano Banana Pro Edit (array of images) ----------
class NanoBananaProEditAPINode:
    """Edit/combine images via Official Google Gemini 3.0 Pro API (Nano Banana Pro)."""

    MODEL_ID = "gemini-3-pro-image-preview"
    BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "aspect_ratio": (["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "4:5", "5:4", "21:9"], {"default": "auto"}),
                "image_size": (["1K", "2K", "4K"], {"default": "2K"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "safety_threshold": (["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"], {"default": "BLOCK_ONLY_HIGH"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "image"

    def edit(self, images, api_key, prompt, aspect_ratio="auto", image_size="2K", temperature=1.0, safety_threshold="BLOCK_ONLY_HIGH"):
        api_key = api_key.strip()
        if not api_key:
            raise RuntimeError("Nano Banana Pro Edit: API key is required.")

        url = f"{self.BASE_URL}?key={api_key}"
        headers = {"Content-Type": "application/json"}

        # 1. Convert input images (tensors) to Base64 parts
        parts = []
        for i in range(images.shape[0]):
            img_np = (255.0 * images[i].cpu().numpy()).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(img_np)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            parts.append({"inlineData": {"mimeType": "image/png", "data": b64}})
        
        # 2. Add the editing prompt
        parts.append({"text": prompt})

        # 3. Build the Payload (Removed person_generation to fix 400 error)
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "temperature": temperature,
                "imageConfig": {
                    "imageSize": image_size
                }
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": safety_threshold},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": safety_threshold},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": safety_threshold},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": safety_threshold}
            ]
        }

        # Handle 'auto' vs specific aspect ratios
        if aspect_ratio != "auto":
            payload["generationConfig"]["imageConfig"]["aspectRatio"] = aspect_ratio

        # 4. Request
        response = requests.post(url, json=payload, headers=headers, timeout=180)
        
        if response.status_code != 200:
            raise RuntimeError(f"Google API Error {response.status_code}: {response.text}")

        data = response.json()

        try:
            # Extract image from Gemini 3 response structure
            img_b64 = data['candidates'][0]['content']['parts'][0]['inlineData']['data']
            img_bytes = base64.b64decode(img_b64)
            output_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            output_np = np.array(output_pil).astype(np.float32) / 255.0
            return (torch.from_numpy(output_np).unsqueeze(0),)
        except (KeyError, IndexError):
            reason = data.get('candidates', [{}])[0].get('finishReason', 'UNKNOWN')
            raise RuntimeError(f"Nano Banana Pro Edit: Failed. Reason: {reason}")


# ---------- Nano Banana 2 (Gemini generateContent) ----------

_NB2_ASPECT_RATIOS = [
    "auto",
    "1:1",
    "2:3", "3:2",
    "3:4", "4:3",
    "9:16", "16:9",
    "21:9",
]

_NB2_IMAGE_SIZES = ["1K", "2K", "4K"]


# ---------- Nano Banana 2 Edit (Gemini generateContent with image input) ----------

class NanoBanana2EditAPINode:
    """Edit images via Nano Banana 2 API (gemini-3.1-flash-image-preview with image input)."""

    MODEL_ID = "gemini-3.1-flash-image-preview"
    BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "aspect_ratio": (_NB2_ASPECT_RATIOS, {"default": "auto"}),
                "image_size": (_NB2_IMAGE_SIZES, {"default": "1K"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "safety_threshold": (["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"], {"default": "BLOCK_ONLY_HIGH"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "image"

    def edit(
        self,
        images: torch.Tensor,
        api_key: str,
        prompt: str,
        aspect_ratio: str = "auto",
        image_size: str = "1K",
        temperature: float = 1.0,
        safety_threshold: str = "BLOCK_ONLY_HIGH",
    ):
        api_key = api_key.strip()
        if not api_key:
            raise RuntimeError("Nano Banana 2 Edit: API key is required.")
        if images.shape[0] == 0:
            raise RuntimeError("Nano Banana 2 Edit: at least one image is required.")

        temperature = temperature if temperature is not None else 1.0

        parts = []
        for i in range(images.shape[0]):
            img_np = (255.0 * images[i].cpu().numpy()).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(img_np)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            parts.append({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": b64,
                }
            })
        parts.append({"text": prompt})

        url = f"{self.BASE_URL}?key={api_key}"
        headers = {"Content-Type": "application/json"}

        image_config = {}
        if aspect_ratio and aspect_ratio != "auto":
            image_config["aspectRatio"] = aspect_ratio
        if image_size and image_size != "auto":
            image_config["imageSize"] = image_size

        gen_config = {
            "responseModalities": ["TEXT", "IMAGE"],
            "temperature": temperature,
        }
        if image_config:
            gen_config["imageConfig"] = image_config

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": gen_config,
        }
        if safety_threshold:
            _apply_gemini_safety(payload, safety_threshold)

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(_api_error_msg("Nano Banana 2 Edit", e))
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Nano Banana 2 Edit request error: {e!s}")

        raw = _extract_image_from_gemini_response(data, "Nano Banana 2 Edit")
        if not raw:
            detail = _gemini_response_summary(data)
            raise RuntimeError(f"Nano Banana 2 Edit: no image in response. {detail}")
        return _image_bytes_to_tensor(raw, "Nano Banana 2 Edit")


# ---------- Shared helpers ----------

def _gemini_response_summary(data: dict) -> str:
    """Return a human-readable summary of why a Gemini response contained no image."""
    try:
        candidates = data.get("candidates") or []
        if not candidates:
            prompt_feedback = data.get("promptFeedback", {})
            block_reason = prompt_feedback.get("blockReason", "")
            if block_reason:
                return f"Prompt was blocked: {block_reason}"
            return "No candidates returned."
        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "")
        parts = candidate.get("content", {}).get("parts") or []
        text_parts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
        if text_parts:
            snippet = " | ".join(text_parts)[:300]
            return f"Model returned text only (finishReason={finish_reason}): {snippet}"
        return f"No image part in response (finishReason={finish_reason})."
    except Exception:
        return ""


def _extract_image_from_gemini_response(data: dict, log_name: str) -> bytes | None:
    """Extract the first non-thought inline image from a Gemini generateContent response."""
    try:
        candidates = data.get("candidates") or []
        if not candidates:
            logging.warning("%s: response has no candidates. Full response: %s", log_name, data)
            return None
        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "")
        parts = candidate.get("content", {}).get("parts") or []
        logging.info("%s: finishReason=%s, part count=%d", log_name, finish_reason, len(parts))
        text_parts = []
        for part in parts:
            if part.get("thought"):
                continue
            if "inlineData" in part:
                b64 = part["inlineData"].get("data")
                if b64:
                    return base64.b64decode(b64)
            if "text" in part:
                text_parts.append(part["text"])
        if text_parts:
            logging.warning(
                "%s: API returned text instead of image (finishReason=%s). Text: %s",
                log_name, finish_reason, " | ".join(text_parts)[:500],
            )
        else:
            logging.warning(
                "%s: no image or text in response. finishReason=%s. Full response: %s",
                log_name, finish_reason, str(data)[:1000],
            )
    except (KeyError, TypeError, ValueError) as e:
        logging.warning("%s parse response: %s — raw: %s", log_name, e, str(data)[:500])
    return None

def _apply_gemini_safety(payload: dict, threshold: str) -> None:
    """Apply safety threshold to Gemini payload (safetySettings)."""
    categories = (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_CIVIC_INTEGRITY",
    )
    payload["safetySettings"] = [
        {"category": cat, "threshold": threshold} for cat in categories
    ]


def _api_error_msg(service: str, e: requests.exceptions.HTTPError) -> str:
    msg = f"{service} API HTTP error: {e.response.status_code}"
    try:
        body = e.response.json()
        if "error" in body:
            msg += f" — {body.get('error', body)}"
        elif "message" in body:
            msg += f" — {body.get('message', body)}"
    except Exception:
        text = e.response.text
        if text:
            msg += f" — {text[:200]}"
    return msg


def _image_bytes_to_tensor(raw: bytes, log_name: str) -> tuple:
    try:
        image = Image.open(io.BytesIO(raw))
        image = image.convert("RGB")
    except Exception as e:
        logging.error("%s: failed to decode image: %s", log_name, e)
        raise RuntimeError(f"Failed to decode image from API: {e}") from e
    arr = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)[None, ...]
    return (tensor,)


def _image_tensor_to_data_uris(images: torch.Tensor) -> list:
    """Convert ComfyUI IMAGE tensor [B, H, W, C] to list of data:image/png;base64,... strings."""
    out = []
    for i in range(images.shape[0]):
        img_np = (255.0 * images[i].cpu().numpy()).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img_np)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        out.append(f"data:image/png;base64,{b64}")
    return out


def _extract_image_from_edit_response(data: dict) -> bytes | None:
    """Extract image bytes from edit API direct response (data[].url or data[].b64_json)."""
    if not isinstance(data, dict):
        return None
    items = data.get("data")
    if not isinstance(items, list) or not items:
        return None
    first = items[0]
    if not isinstance(first, dict):
        return None
    url = first.get("url")
    if url:
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.content
        except requests.exceptions.RequestException:
            return None
    b64 = first.get("b64_json")
    if b64:
        try:
            return base64.b64decode(b64)
        except (ValueError, TypeError):
            return None
    return None


# ---------- ComfyUI registration ----------

NODE_CLASS_MAPPINGS = {
    "NanoBananaProEditAPINode": NanoBananaProEditAPINode,
    "NanoBanana2EditAPINode": NanoBanana2EditAPINode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaProEditAPINode": "Nano Banana Pro Edit",
    "NanoBanana2EditAPINode": "Nano Banana 2 Edit",
}
