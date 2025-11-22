from __future__ import annotations

import json
from dataclasses import dataclass, asdict
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI


# ---------- 1. Общее состояние пайплайна ----------


class AppState(TypedDict, total=False):
    # Вход
    file_path: str
    file_ext: str
    file_bytes: bytes

    # Текст документа
    raw_text: str
    doc_type: str  # "questionnaire" / "spec" / "other"

    # Структурированная заявка
    request_fields: Dict[str, Any]      # как вернул LLM
    matched_items: List[Dict[str, Any]] # найденные позиции в каталоге
    export_payload: Dict[str, Any]

    # Лог шагов для UI
    messages: List[str]


# ---------- 2. Модель заявки (нейтральная) ----------


class RequestFieldsModel(BaseModel):
    """Что хотим вытащить из заявки (общая схема)."""

    customer_name: Optional[str] = Field(
        None, description="Имя или организация клиента"
    )
    project_name: Optional[str] = Field(
        None, description="Название проекта / объекта"
    )
    product_type: Optional[str] = Field(
        None, description="Тип изделия, например 'задвижка', 'муфта', 'кран'"
    )
    medium: Optional[str] = Field(
        None, description="Транспортируемая среда, например 'газ', 'нефть'"
    )
    pressure: Optional[str] = Field(
        None, description="Давление (строка из заявки: МПа, кгс/см2 и т.п.)"
    )
    diameter: Optional[str] = Field(
        None, description="Диаметр / типоразмер, например 'DN50', '159x6'"
    )
    temperature: Optional[str] = Field(
        None, description="Температура среды или диапазон"
    )
    installation_type: Optional[str] = Field(
        None, description="Место установки, например 'подземная', 'надземная'"
    )
    connection_type: Optional[str] = Field(
        None, description="Тип присоединения: фланцевое, сварное, резьбовое и т.п."
    )
    coatings: Optional[str] = Field(
        None, description="Требования к покрытию (внутреннее/наружное)"
    )
    climate: Optional[str] = Field(
        None, description="Климатическое исполнение или диапазон температур окружающей среды"
    )
    quantity: Optional[int] = Field(
        None, description="Количество единиц продукции"
    )
    extra_requirements: Optional[str] = Field(
        None, description="Дополнительные требования и комментарии"
    )


@dataclass
class RequestFields:
    customer_name: Optional[str] = None
    project_name: Optional[str] = None
    product_type: Optional[str] = None
    medium: Optional[str] = None
    pressure: Optional[str] = None
    diameter: Optional[str] = None
    temperature: Optional[str] = None
    installation_type: Optional[str] = None
    connection_type: Optional[str] = None
    coatings: Optional[str] = None
    climate: Optional[str] = None
    quantity: Optional[int] = None
    extra_requirements: Optional[str] = None


# ---------- 3. LLM (LangChain) ----------

# OPENAI_API_KEY берётся из ENV
load_dotenv() 
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
if not api_key:
    raise RuntimeError(
        "Не найден API-ключ. Установи переменную окружения OPENAI_API_KEY "
        "или GPT_API_KEY перед запуском."
    )

llm = ChatOpenAI(
    model="gpt-4o-mini",   # или другую модель, которую ты используешь
    temperature=0.1,
    api_key=api_key,
)

structured_llm = llm.with_structured_output(RequestFieldsModel)


# ---------- 4. Утилиты (OCR/парсинг — пока заглушки) ----------


def add_msg(state: AppState, text: str) -> None:
    msgs = state.get("messages") or []
    msgs.append(text)
    state["messages"] = msgs


def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def extract_text_from_file(path: str, ext: str, data: bytes) -> str:
    """
    Сюда потом можно вкрутить реальный OCR+парсинг:
    - .docx → python-docx
    - .xlsx → pandas/openpyxl
    - .pdf/.jpg/.png → внешний OCR-сервис (ключи в ENV).
    Сейчас — stub.
    """
    return f"(stub) text extracted from file {Path(path).name} (ext={ext}), size={len(data)} bytes"


def classify_document(text: str) -> str:
    """
    LLM-классификация: опросный лист / тех.описание / другое.
    """
    messages = [
        SystemMessage(
            content=(
                "Ты классифицируешь тип инженерного документа. "
                "Если документ похож на опросный лист заказчика "
                "(таблица параметров, поля для заполнения, запрос на подбор оборудования), "
                "ответь 'questionnaire'. Если это техническое описание продукции "
                "или ТУ, ответь 'spec'. Иначе ответь 'other'."
            )
        ),
        HumanMessage(content=text[:4000]),
    ]
    resp = llm.invoke(messages)
    ans = resp.content.strip().lower()
    if "questionnaire" in ans:
        return "questionnaire"
    if "spec" in ans:
        return "spec"
    return "other"


def match_with_catalog(fields: RequestFields) -> List[Dict[str, Any]]:
    """
    Заглушка сопоставления с базой ТУ / 1С.
    Потом здесь можно делать SQL/REST в 1С или поиск по своему каталогу.
    Сейчас просто возвращаем один фиктивный вариант.
    """
    return [
        {
            "item_code": "ITEM-001",
            "name": f"Подходящее изделие для {fields.product_type or 'изделия'}",
            "score": 0.8,
            "matched_fields": {
                "product_type": fields.product_type,
                "medium": fields.medium,
                "pressure": fields.pressure,
                "diameter": fields.diameter,
            },
        }
    ]


def build_export_payload(fields: RequestFields, matched: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Структура, которую можно отправить в 1С / записать в Excel / CSV.
    """
    return {
        "request_fields": asdict(fields),
        "matched_items": matched,
    }


# ---------- 5. Узлы (агенты) LangGraph ----------


def file_ingestion_node(state: AppState) -> AppState:
    path = state.get("file_path")
    if not path:
        raise ValueError("file_path is missing in state")

    ext = Path(path).suffix.lower()
    data = read_file_bytes(path)

    state["file_ext"] = ext
    state["file_bytes"] = data
    add_msg(state, f"[file_ingestion] Loaded file {Path(path).name} (ext={ext}, size={len(data)} bytes).")
    return state


def text_extraction_node(state: AppState) -> AppState:
    path = state.get("file_path")
    ext = state.get("file_ext")
    data = state.get("file_bytes")

    if not path or ext is None or data is None:
        raise ValueError("file_path/file_ext/file_bytes not set for text extraction")

    text = extract_text_from_file(path, ext, data)
    state["raw_text"] = text
    add_msg(state, f"[text_extraction] Extracted text of length {len(text)} chars.")
    return state


def doc_classifier_node(state: AppState) -> AppState:
    text = state.get("raw_text", "")
    doc_type = classify_document(text)
    state["doc_type"] = doc_type
    add_msg(state, f"[doc_classifier] Document type: {doc_type}.")
    return state


def field_extraction_node(state: AppState) -> AppState:
    if state.get("doc_type") != "questionnaire":
        add_msg(state, "[field_extraction] Document is not a questionnaire, skipping extraction.")
        return state

    text = state.get("raw_text", "")

    system_msg = SystemMessage(
        content=(
            "Ты извлекаешь параметры заявки клиента на техническое оборудование "
            "в структурированный формат. Ничего не выдумывай: если параметра нет, "
            "оставляй null."
        )
    )
    user_msg = HumanMessage(
        content=(
            "Вот текст заявки. Заполни схему RequestFieldsModel.\n\n"
            + text[:6000]
        )
    )

    result: RequestFieldsModel = structured_llm.invoke([system_msg, user_msg])
    fields = RequestFields(**result.dict())
    state["request_fields"] = asdict(fields)
    add_msg(state, "[field_extraction] Extracted request fields: " + json.dumps(asdict(fields), ensure_ascii=False))
    return state


def matching_node(state: AppState) -> AppState:
    if "request_fields" not in state:
        add_msg(state, "[matching] No request_fields in state, nothing to match.")
        return state

    fields = RequestFields(**state["request_fields"])
    items = match_with_catalog(fields)
    state["matched_items"] = items
    add_msg(state, "[matching] Found catalog matches: " + json.dumps(items, ensure_ascii=False))
    return state


def export_node(state: AppState) -> AppState:
    fields_dict = state.get("request_fields", {})
    items = state.get("matched_items", [])

    fields = RequestFields(**fields_dict) if fields_dict else RequestFields()
    payload = build_export_payload(fields, items)
    state["export_payload"] = payload
    add_msg(state, "[export] Built export payload.")
    return state


# ---------- 6. Сборка графа ----------


def build_processing_graph():
    workflow = StateGraph(AppState)

    workflow.add_node("file_ingestion", file_ingestion_node)
    workflow.add_node("text_extraction", text_extraction_node)
    workflow.add_node("doc_classifier", doc_classifier_node)
    workflow.add_node("field_extraction", field_extraction_node)
    workflow.add_node("matching", matching_node)
    workflow.add_node("export", export_node)

    workflow.set_entry_point("file_ingestion")

    workflow.add_edge("file_ingestion", "text_extraction")
    workflow.add_edge("text_extraction", "doc_classifier")
    workflow.add_edge("doc_classifier", "field_extraction")
    workflow.add_edge("field_extraction", "matching")
    workflow.add_edge("matching", "export")
    workflow.add_edge("export", END)

    return workflow.compile()


# ---------- 7. Локальный тест ----------

if __name__ == "__main__":
    graph = build_processing_graph()
    example_path = "uploads/example_request.pdf"  # подставь свой путь

    init_state: AppState = {
        "file_path": example_path,
        "messages": [],
    }

    final_state = graph.invoke(init_state)
    final_state.pop("file_bytes", None)

    print("=== FINAL STATE ===")
    print(json.dumps(final_state, ensure_ascii=False, indent=2))

    print("\n=== LOG ===")
    for m in final_state.get("messages", []):
        print(m)
