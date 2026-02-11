import streamlit as st
import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from sqlalchemy import create_engine, text

# =======================
# STREAMLIT UI
# =======================
st.set_page_config(page_title="Hospital AI Assistant", layout="centered")
st.title("🏥 AI-помічник лікарні")

# =======================
# ЗАВАНТАЖЕННЯ КЛЮЧІВ
# =======================
load_dotenv()

gemini_api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
pinecone_api_key = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")

USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

if not gemini_api_key or not pinecone_api_key:
    st.error("❌ Не знайдено GEMINI_API_KEY або PINECONE_API_KEY")
    st.stop()

# =======================
# LLM (БЕЗ ЗМІН)
# =======================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=gemini_api_key,
)

FORBIDDEN_SQL = [
    "insert", "update", "delete", "drop",
    "truncate", "alter", "create", "grant", "revoke"
]


def clean_sql(sql: str) -> str:
    sql = sql.replace("```sql", "").replace("```", "")
    return sql.strip()


# =======================
# ВЕКТОРНА БАЗА (БЕЗ ЗМІН)
# =======================
def search_vector_db(query: str, k=3):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=gemini_api_key
    )

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("hospital-docs")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    results = vector_store.similarity_search(query, k=k)

    vector_texts = []
    for res in results:
        vector_texts.append(
            f"Блок: {res.metadata.get('section_title', 'NO BLOCK TITLE')}\n"
            f"Файл: {res.metadata.get('file_name')}\n"
            f"{res.page_content[:500]}"
        )

    return "\n\n".join(vector_texts) if vector_texts else "Релевантних документів не знайдено."


# =======================
# РЕЛЯЦІЙНА БАЗА (БЕЗ ЗМІН)
# =======================
def query_relational_db(query: str):
    DATABASE_URL = (
        f"postgresql+psycopg2://{USER}:{PASSWORD}"
        f"@{HOST}:{PORT}/{DBNAME}?sslmode=require"
    )
    engine = create_engine(DATABASE_URL)

    system_msg = SystemMessage(
        content=(
            "Ти агент для PostgreSQL бази даних. Тобі дозволено лише читати дані. "
            "Будь-які зміни заборонені. У базі даних є таблиці та колонки: "
            "Departments(ID, BUILDING, FINANCING, NAME), Diseases(ID, NAME, SEVERITY), "
            "Doctors(ID, NAME, PHONE, SALARY, SURNAME), Examinations(ID, NAME, DAYOFWEEK, STARTTIME, ENDTIME), "
            "Wards(ID, DEPARTMENT, BUILDING, FLOOR, NAME), Specializations(ID, NAME), "
            "DoctorsSpecializations(ID, DOCTOR_ID, SPECIALIZATION_ID), Sponsors(ID, NAME), "
            "Donations(ID, AMOUNT, DONATION_DATE, DEPARTMENT_ID, SPONSOR_ID), "
            "Vacations(ID, STARTDATE, ENDDATE, DOCTOR_ID)."
        )
    )

    human_msg = HumanMessage(
        content=f"""
        Створи SQL-запит для цього запиту користувача:

        {query}

        Відповідь має:
        - Бути лише читальною SQL-командою (SELECT, WITH, EXPLAIN)
        - Не містити INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, CREATE
        - Поверни лише SQL без додаткового тексту
        """
    )

    response = llm.generate([[system_msg, human_msg]])
    sql_query = clean_sql(response.generations[0][0].text).lower()

    if any(word in sql_query for word in FORBIDDEN_SQL):
        return "Помилка: агент не може змінювати базу даних."

    with engine.connect() as conn:
        result = conn.execute(text(sql_query))
        rows = result.fetchall()
        columns = result.keys()

    if not rows:
        return "На жаль, результатів за вашим запитом не знайдено."

    table = [dict(zip(columns, row)) for row in rows]

    data_msg = HumanMessage(
        content=f"Використовуючи ці дані, створи ввічливу природну відповідь:\n{table}"
    )

    return llm.generate([[data_msg]]).generations[0][0].text.strip()


# =======================
# ГІБРИДНИЙ АГЕНТ (БЕЗ ЗМІН)
# =======================
def hybrid_agent(query: str):
    vector_text = search_vector_db(query)
    relational_text = query_relational_db(query)

    final_prompt = HumanMessage(
        content=f"""
        ДАВАЙ МАКСИМАЛЬНО КОРОТКУ ТА ЧІТКУ ВІДПОВІДЬ ВВІЧЛИВО!
        ПРО ІНФОРМАЦІЙНІ СИСТЕМИ ТИ НІЧОГО НЕ ПИШЕШ, ЯКЩО НЕ ЗАПИТУЮ ПРО НЕЇ!

        1️⃣ Реляційна база :
        {relational_text}

        2️⃣ Векторна база :
        {vector_text}

        Перед відповіддю користувачу, проаналізуй обидва джерела та створи єдину,
        коротку та людську відповідь.
        """
    )

    response = llm.generate([[final_prompt]])
    return response.generations[0][0].text.strip()


# =======================
# ІСТОРІЯ ЧАТУ (ЯК НА ПРАКТИЦІ)
# =======================
if "history" not in st.session_state:
    st.session_state["history"] = []

user_query = st.chat_input("Введіть ваше питання до лікарні")

if user_query:
    st.session_state["history"].append(HumanMessage(user_query))

    with st.spinner("Думаю..."):
        answer = hybrid_agent(user_query)

    st.session_state["history"].append(AIMessage(answer))

# =======================
# ВІДОБРАЖЕННЯ ІСТОРІЇ
# =======================
for msg in st.session_state["history"]:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
