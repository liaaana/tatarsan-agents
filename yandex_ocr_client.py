# yandex_ocr_client.py
import base64
import json
import os
from pathlib import Path
from typing import Sequence

import requests


YANDEX_OCR_URL = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"


class YandexOcrError(Exception):
    """Ошибка при вызове Яндекс Vision OCR."""


def _get_api_key() -> str:
    """
    Берём API-ключ из окружения.

    Поддерживаем два варианта:
    - YANDEX_OCR_API_KEY
    - OCR_API_KEY (если ты уже его используешь)
    """
    api_key = os.getenv("YANDEX_OCR_API_KEY") or os.getenv("OCR_API_KEY")
    if not api_key:
        raise YandexOcrError(
            "Не настроен ключ OCR. Задай YANDEX_OCR_API_KEY "
            "или OCR_API_KEY в .env / переменных окружения."
        )
    return api_key


def _guess_mime_type(path: Path) -> str:
    """
    Возвращаем mimeType в том формате, как ждёт Яндекс OCR:
    'JPEG', 'PNG', 'PDF'
    """
    s = path.suffix.lower()
    if s in (".jpg", ".jpeg"):
        return "JPEG"
    if s == ".png":
        return "PNG"
    if s == ".pdf":
        return "PDF"
    raise YandexOcrError(f"Тип файла не поддерживается OCR: {path.name}")


def recognize_file_to_text(
    file_path: str,
    language_codes: Sequence[str] = ("ru", "en"),
    model: str = "page",        # базовая модель распознавания страниц
    timeout: float = 40.0,
) -> str:
    """
    Синхронный запрос к Vision OCR: /ocr/v1/recognizeText

    Возвращает полный текст (fullText) одной страницы (как есть).
    """
    path = Path(file_path)
    if not path.exists():
        raise YandexOcrError(f"Файл не найден: {file_path}")

    api_key = _get_api_key()
    mime_type = _guess_mime_type(path)

    # кодируем файл в Base64
    with path.open("rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "mimeType": mime_type,
        "languageCodes": list(language_codes),
        "model": model,
        "content": content_b64,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {api_key}",
    }

    resp = requests.post(
        YANDEX_OCR_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        text = resp.text
        raise YandexOcrError(
            f"OCR HTTP error: {e} / body={text[:500]!r}"
        ) from e

    data = resp.json()

    # Поддерживаем варианты:
    # { "textAnnotation": {...} } или { "result": { "textAnnotation": {...} } }
    text_annotation = data.get("textAnnotation")
    if not text_annotation and "result" in data:
        text_annotation = data["result"].get("textAnnotation")

    if not text_annotation:
        raise YandexOcrError(
            f"В ответе нет textAnnotation: {json.dumps(data)[:500]}"
        )

    full_text = (text_annotation.get("fullText") or "").strip()
    return full_text
