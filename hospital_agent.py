import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from sqlalchemy import create_engine, text

# -----------------------
# Завантаження ключів
# -----------------------
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

if not gemini_api_key or not pinecone_api_key:
    raise ValueError("Не знайдено GEMINI_API_KEY або PINECONE_API_KEY у .env файлі")

# -----------------------
# LLM
# -----------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=gemini_api_key,
)

FORBIDDEN_SQL = ["insert", "update", "delete", "drop", "truncate", "alter", "create", "grant", "revoke"]

def clean_sql(sql: str) -> str:
    sql = sql.replace("```sql", "").replace("```", "")
    return sql.strip()

# =======================
# Функція роботи з векторною базою
# =======================
def search_vector_db(query: str, k=3):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=gemini_api_key
    )

    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "hospital-docs"
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    vector_results = vector_store.similarity_search(query, k=k)
    vector_texts = []
    for res in vector_results:
        vector_texts.append(f"Блок: {res.metadata.get('section_title', 'NO BLOCK TITLE')}\n"
                            f"Файл: {res.metadata.get('file_name')}\n"
                            f"{res.page_content[:500]}")  # обмеження 500 символів

    return "\n\n".join(vector_texts) if vector_texts else "Релевантних документів не знайдено."

# =======================
# Функція роботи з реляційною базою
# =======================
def query_relational_db(query: str):
    DATABASE_URL = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}?sslmode=require"
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

    if any(keyword in sql_query for keyword in FORBIDDEN_SQL):
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
# Гібридна функція
# =======================
def hybrid_agent(query: str):
    # 1️⃣ Векторна база
    vector_text = search_vector_db(query)

    # 2️⃣ Реляційна база
    relational_text = query_relational_db(query)

    # 3️⃣ Об’єднання обох джерел у фінальну відповідь через LLM
    final_prompt = HumanMessage(
        content=f"""
        ДАВАЙ МАКСИМАЛЬНО КОРОТКУ ТА ЧІТКУ ВІДПОВІДЬ ВВІЧЛИВО!
        ПРО ІНФОРМАЦІЙНІ СИСТЕМИ ТИ НІЧОГО НЕ ПИШЕШ, ЯКЩО НЕ ЗАПИТУЮ ПРО НЕЇ!
        У тебе є два джерела інформації для відповіді користувачу на запит: 
        

        
        1️⃣ Реляційна база :
        {relational_text}
        
        2️⃣ Векторна база :
        {vector_text}
        
        Перед відповіддю користувачу, проаналізуй обидва джерела та вибери і створи єдину зрозумілу, 
        чітку, коротку та ввічливу відповідь користувачу, використовуючи обидва джерела.
        Відповідь має бути максимально інформативною, людською мовою, без зайвої інформації.
        """
    )

    final_response = llm.generate([[final_prompt]])
    return final_response.generations[0][0].text.strip()

# =======================
# Приклад використання
# =======================
if __name__ == "__main__":
    query = "в які години можна попасти до кардіолога?"
    answer = hybrid_agent(query)
    print(answer)
