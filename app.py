import base64
import csv
import datetime
import io
import json
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from processing_graph import build_processing_graph, RequestFields

# Компилируем граф один раз при старте
GRAPH = build_processing_graph()


# ---------- Утилиты ----------

def ensure_dirs():
    Path("uploads").mkdir(exist_ok=True)


def save_uploaded_file(uploaded, dest_dir: str = "uploads") -> str:
    ensure_dirs()
    suffix = Path(uploaded.name).suffix
    name = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"
    path = Path(dest_dir) / name
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return str(path)


def save_submission(image_path: str, fields: list[str], csv_path: str = "submissions.csv") -> None:
    header = ["timestamp", "image"] + [f"field_{i+1}" for i in range(11)] + ["joined"]
    exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        row = [
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            image_path,
        ] + fields + [" - ".join(fields)]
        writer.writerow(row)


def read_submissions(csv_path: str = "submissions.csv") -> list[dict]:
    if not Path(csv_path).exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def request_fields_to_ui_list(fields_dict: dict) -> list[str]:
    """Маппим RequestFields → 11 строк для полей UI."""
    rf = RequestFields(**fields_dict) if fields_dict else RequestFields()
    res: list[str] = [
        rf.customer_name or "",            # 1
        rf.project_name or "",             # 2
        rf.product_type or "",             # 3
        rf.medium or "",                   # 4
        rf.pressure or "",                 # 5
        rf.diameter or "",                 # 6
        rf.temperature or "",              # 7
        rf.installation_type or "",        # 8
        rf.connection_type or "",          # 9
        rf.coatings or "",                 # 10
        (rf.climate or "") + (f", qty={rf.quantity}" if rf.quantity is not None else ""),  # 11
    ]
    return res


# ---------- Основное приложение ----------

def main() -> None:
    st.set_page_config(page_title="Заявки клиентов", page_icon="🏛️", layout="wide")

    # Лёгкий CSS-твик
    st.markdown(
        """
        <style>
        :root {
          --primary: #027fa9;
          --primary-dark: #026a86;
          --primary-light: #79cde0;
          --text-dark: #083a45;
        }

        .stApp {
            background: linear-gradient(180deg, #f0fbff 0%, #ffffff 100%);
            color: var(--text-dark);
            font-family: 'Helvetica Neue', Arial, sans-serif;
        }

        .stButton>button, .stDownloadButton>button {
            background-color: var(--primary) !important;
            color: white !important;
            border-radius: 6px;
            border: none;
            padding: 8px 12px;
        }
        .stButton>button:hover, .stDownloadButton>button:hover {
            background-color: var(--primary-dark) !important;
        }

        .stFileUploader, .stFileUploader div {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Загрузка заявок и полуавтоматическая обработка (AI)")

    tab_upload, tab_dashboard = st.tabs(["Форма загрузки", "Дашборд"])

    # ------------- Вкладка: Форма загрузки -------------

    with tab_upload:
        st.header("1️⃣ Загрузите файл заявки")
        uploaded = st.file_uploader(
            "Выберите файл (png/jpg/jpeg/pdf/docx/xlsx)",
            type=["png", "jpg", "jpeg", "pdf", "docx", "xlsx"],
        )

        saved_path = None
        if uploaded is not None:
            try:
                saved_path = save_uploaded_file(uploaded)
                st.session_state["last_uploaded_path"] = saved_path
                suffix = Path(saved_path).suffix.lower()

                # Предпросмотр
                if suffix in [".png", ".jpg", ".jpeg"]:
                    st.image(saved_path, use_container_width=True)
                elif suffix == ".pdf":
                    try:
                        with open(saved_path, "rb") as f:
                            pdf_bytes = f.read()
                        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                        pdf_display = f"<iframe src='data:application/pdf;base64,{b64}' width='100%' height='600' style='border: none;'></iframe>"
                        components.html(pdf_display, height=600)
                    except Exception:
                        st.info(f"Загружен PDF: {Path(saved_path).name}")
                elif suffix == ".docx":
                    try:
                        from docx import Document
                        doc = Document(saved_path)
                        text = "\n".join([p.text for p in doc.paragraphs if p.text])
                        if text:
                            st.text_area(
                                "Предпросмотр DOCX (первые 2000 символов):",
                                value=text[:2000],
                                height=300,
                            )
                        else:
                            st.info(f"Загружен DOCX: {Path(saved_path).name}")
                    except Exception:
                        st.info(f"Загружен DOCX: {Path(saved_path).name}")
                elif suffix in [".xls", ".xlsx"]:
                    try:
                        df_preview = pd.read_excel(saved_path, nrows=20)
                        st.dataframe(df_preview, use_container_width=True)
                    except Exception:
                        st.info(f"Загружен Excel: {Path(saved_path).name}")
                else:
                    st.info(f"Загружен файл: {Path(saved_path).name}")
                    with open(saved_path, "rb") as f:
                        st.download_button(
                            "Скачать загруженный файл",
                            data=f,
                            file_name=Path(saved_path).name,
                        )

                st.success(f"Файл сохранён: {saved_path}")
            except Exception as e:
                st.error(f"Не удалось сохранить файл: {e}")

        st.write("---")
        st.subheader("2️⃣ Параметры заявки (11 полей)")

        st.write(
            "AI будет пытаться заполнить их автоматически из файла, "
            "но вы можете отредактировать руками."
        )

        # Инициализация 11 полей
        fields: list[str] = []
        cols = st.columns(11)
        labels = [
            "Клиент",
            "Проект",
            "Тип изделия",
            "Среда",
            "Давление",
            "Диаметр",
            "Температура",
            "Установка",
            "Присоединение",
            "Покрытия",
            "Климат / qty",
        ]
        for i, col in enumerate(cols):
            key = f"field_{i+1}"
            st.session_state.setdefault(key, "")
            fields.append(
                col.text_input(
                    labels[i],
                    key=key,
                    placeholder="(пусто)",
                )
            )

        st.write("**Общее (через —):**")
        st.write(
            " - ".join(
                [
                    v
                    for v in [
                        st.session_state.get(f"field_{i+1}", "") for i in range(11)
                    ]
                    if v
                ]
            )
        )

        # Кнопка AI-заполнения
        st.write("---")
        st.subheader("3️⃣ Заполнить поля с помощью AI (LangGraph)")

        if "processing_error" in st.session_state:
            st.error(st.session_state["processing_error"])
            # очищаем после показа
            st.session_state.pop("processing_error", None)

        def _fill_from_ai() -> None:
            path = st.session_state.get("last_uploaded_path")
            if not path:
                st.session_state["processing_error"] = "Сначала загрузите файл заявки."
                return

            try:
                state = {
                    "file_path": path,
                    "messages": [],
                }
                result = GRAPH.invoke(state)
                result.pop("file_bytes", None)
                st.session_state["processing_result"] = result

                rf_dict = result.get("request_fields") or {}
                ui_vals = request_fields_to_ui_list(rf_dict)
                for i in range(11):
                    st.session_state[f"field_{i+1}"] = ui_vals[i]
            except Exception as e:
                st.session_state["processing_error"] = f"Ошибка при обработке: {e}"

        st.button(
            "Заполнить поля из файла (AI)",
            on_click=_fill_from_ai,
            disabled=("last_uploaded_path" not in st.session_state),
        )

        if "processing_result" in st.session_state:
            st.markdown("**Результат мультиагентной обработки:**")
            st.json(st.session_state["processing_result"])

        # Отправка заявки
        st.write("---")
        st.subheader("4️⃣ Сохранить заявку")

        if st.button("Отправить заявку"):
            # берём последний сохранённый путь или из сессии
            img_path = saved_path or st.session_state.get("last_uploaded_path")
            if not img_path:
                st.error("Пожалуйста, загрузите файл перед отправкой.")
            else:
                try:
                    collected = [
                        st.session_state.get(f"field_{i+1}", "") for i in range(11)
                    ]
                    collected = [v if v is not None else "" for v in collected]
                    save_submission(img_path, collected)
                    st.success("Заявка сохранена.")
                    # очистим поля для следующей
                    for i in range(11):
                        st.session_state.pop(f"field_{i+1}", None)
                except Exception as e:
                    st.error(f"Ошибка при сохранении заявки: {e}")

    # ------------- Вкладка: Дашборд -------------

    with tab_dashboard:
        st.header("Дашборд заявок")
        submissions = read_submissions()
        st.write(f"Всего заявок: {len(submissions)}")

        if submissions:
            df = pd.DataFrame(submissions)

            if "timestamp" in df.columns:
                df["_ts_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
                valid_ts = df["_ts_parsed"].dropna()
                if not valid_ts.empty:
                    min_date = valid_ts.min().date()
                    max_date = valid_ts.max().date()
                else:
                    min_date = max_date = None
            else:
                df["_ts_parsed"] = pd.NaT
                min_date = max_date = None

            st.write("**Фильтр по дате (UTC):**")
            col_from, col_to = st.columns(2)
            with col_from:
                start_date = st.date_input(
                    "От", value=min_date or datetime.date.today(), key="filter_from"
                )
            with col_to:
                end_date = st.date_input(
                    "До", value=max_date or datetime.date.today(), key="filter_to"
                )

            try:
                if min_date is not None:
                    mask = (df["_ts_parsed"].dt.date >= start_date) & (
                        df["_ts_parsed"].dt.date <= end_date
                    )
                    filtered = df[mask].copy()
                else:
                    filtered = df.copy()
            except Exception:
                filtered = df.copy()

            to_show = filtered.drop(
                columns=[c for c in ["_ts_parsed"] if c in filtered.columns]
            )
            st.dataframe(to_show, use_container_width=True)

            try:
                towrite = io.BytesIO()
                to_show.to_excel(
                    towrite,
                    index=False,
                    engine="openpyxl",
                )
                towrite.seek(0)
                st.download_button(
                    "Скачать Excel (.xlsx)",
                    data=towrite,
                    file_name="submissions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.warning(f"Не удалось сформировать Excel: {e}")
        else:
            st.info("Заявок ещё нет. Отправьте первую через вкладку 'Форма загрузки'.")


if __name__ == "__main__":
    main()
