# ==============РОБОЧИЙ КОД ДЛЯ ЗАВАНТАЖЕННЯ ДОКУМЕНТІВ У PINECONE================

# -*- coding: utf-8 -*-
import os
import json
import dotenv
from uuid import uuid4
from datetime import datetime
from docx import Document
from pypdf import PdfReader

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document as LC_Document

# -----------------------
# Завантаження ключів
# -----------------------
dotenv.load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not gemini_api_key or not pinecone_api_key:
    raise ValueError("Не знайдено GEMINI_API_KEY або PINECONE_API_KEY у .env файлі")

# -----------------------
# Ініціалізація ембедінгів
# -----------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=gemini_api_key
)

# -----------------------
# Ініціалізація Pinecone
# -----------------------
index_name = "hospital-docs"
dimension = 3072

pc = Pinecone(api_key=pinecone_api_key)

# Використовуємо твій стиль з has_index і ServerlessSpec
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# -----------------------
# Функції для парсингу DOCX
# -----------------------
def parse_docx_full_sections(path):
    doc = Document(path)
    sections = []
    current_title = None
    current_text = []

    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        if p.style.name.startswith("Heading"):
            if current_title:
                sections.append({
                    "section_title": current_title,
                    "content": "\n".join(current_text)
                })
            current_title = text
            current_text = []
        else:
            current_text.append(text)

    if current_title:
        sections.append({
            "section_title": current_title,
            "content": "\n".join(current_text)
        })
    return sections


# -----------------------
# Функції для парсингу PDF
# -----------------------
def parse_pdf_full_sections(path):
    reader = PdfReader(path)
    lines = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            lines.extend([line.strip() for line in text.split("\n") if line.strip()])

    sections = []
    n = len(lines)
    i = 0

    # Заголовок документа
    if n > 0:
        sections.append({
            "section_title": "Заголовок документа",
            "content": lines[0]
        })
        i = 1

    # ЗМІСТ до першого основного розділу
    toc_lines = []
    while i < n and not (len(lines[i]) >= 2 and lines[i][0].isdigit() and lines[i][1] == '.' and (
            len(lines[i]) == 2 or lines[i][2] == ' ')):
        toc_lines.append(lines[i])
        i += 1
    if toc_lines:
        sections.append({
            "section_title": "ЗМІСТ",
            "content": "\n".join(toc_lines)
        })

    # Основні розділи
    current_title = None
    current_text = []
    while i < n:
        line = lines[i]
        if len(line) >= 3 and line[0].isdigit() and line[1] == '.' and line[2] == ' ':
            if current_title:
                sections.append({
                    "section_title": current_title,
                    "content": "\n".join(current_text)
                })
            current_title = line
            current_text = []
        else:
            current_text.append(line)
        i += 1

    if current_title:
        sections.append({
            "section_title": current_title,
            "content": "\n".join(current_text)
        })

    return sections


# -----------------------
# Функція завантаження секцій у Pinecone
# -----------------------
def upload_sections(sections, filename):
    docs = []
    now = datetime.utcnow().isoformat()

    for s in sections:
        text = s["content"]
        if not text.strip():
            continue
        doc = LC_Document(
            page_content=text,
            metadata={
                "file_name": filename,
                "section_title": s["section_title"],
                "created_at": now
            }
        )
        docs.append(doc)

    ids = [str(uuid4()) for _ in range(len(docs))]

    # Зберігаємо ID у JSON
    json_path = "ids.json"
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            id_map = json.load(f)
    else:
        id_map = {}

    for doc, id_ in zip(docs, ids):
        key = f'{doc.metadata["file_name"]}::{doc.metadata["section_title"]}'
        id_map[key] = id_

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(id_map, f, indent=2, ensure_ascii=False)

    # Завантажуємо у Pinecone
    vector_store.add_documents(documents=docs, ids=ids)

    return ids


# -----------------------
# Основний запуск
# -----------------------
docx_sections = parse_docx_full_sections("ForWorkers.docx")
pdf_sections = parse_pdf_full_sections("General.pdf")

docx_ids = upload_sections(docx_sections, "ForWorkers.docx")
pdf_ids = upload_sections(pdf_sections, "General.pdf")

print("Документи завантажено у Pinecone")
print("Перші 5 ID DOCX:", docx_ids[:5])
print("Перші 5 ID PDF:", pdf_ids[:5])






# # =============РОБОЧИЙ КОД ДЛЯ ПЕРЕВІРКИ АГЕНТА================
#
# import os
# import dotenv
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from pinecone import Pinecone
# from langchain_pinecone import PineconeVectorStore
#
# # -----------------------
# # Завантаження ключів
# # -----------------------
# dotenv.load_dotenv()
# gemini_api_key = os.getenv("GEMINI_API_KEY")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")
#
# if not gemini_api_key or not pinecone_api_key:
#     raise ValueError("Не знайдено GEMINI_API_KEY або PINECONE_API_KEY у .env файлі")
#
# # -----------------------
# # Підключення ембедінгів і Pinecone
# # -----------------------
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-004",
#     google_api_key=gemini_api_key
# )
#
# pc = Pinecone(api_key=pinecone_api_key)
# index_name = "hospital-docs"
#
# # Підключаємося до існуючого індексу
# index = pc.Index(index_name)
# vector_store = PineconeVectorStore(index=index, embedding=embeddings)
#
# # -----------------------
# # Перевірка агента
# # -----------------------
# query = "Коли обідня перерва та який робочий час у лікарні?"  # приклад запиту
#
# results = vector_store.similarity_search(query, k=3)
#
# print("\n===== Результати перевірки агента =====")
# for res in results:
#     print("BLOCK:", res.metadata.get("section_title", "NO BLOCK TITLE"))
#     print("FILE:", res.metadata.get("file_name"))
#     print("CREATED AT:", res.metadata.get("created_at"))
#     print(res.page_content[:300])  # перші 300 символів тексту
#     print("-" * 60)


