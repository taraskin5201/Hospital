from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, SystemMessage

# --- Завантаження змінних оточення ---
load_dotenv()
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# --- Підключення до бази даних ---
DATABASE_URL = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}?sslmode=require"
engine = create_engine(DATABASE_URL)

# --- LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=gemini_api_key,
)

# --- Список заборонених ключових слів ---
FORBIDDEN_SQL = ["insert", "update", "delete", "drop", "truncate", "alter", "create", "grant", "revoke"]

# --- Функція для очищення SQL від Markdown ---
def clean_sql(sql: str) -> str:
    sql = sql.replace("```sql", "").replace("```", "")
    return sql.strip()

# --- Read-only агент ---
def run_read_only_agent(natural_query: str):
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
        
        {natural_query}
        
        Відповідь має:
        - Бути лише читальною SQL-командою (SELECT, WITH, EXPLAIN)
        - Не містити INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, CREATE
        - Поверни лише SQL без додаткового тексту
        """
            )

    # Генерація SQL через LLM
    response = llm.generate([[system_msg, human_msg]])
    sql_query = response.generations[0][0].text
    sql_query_clean = clean_sql(sql_query).lower()

    # Перевірка на заборонені команди
    if any(keyword in sql_query_clean for keyword in FORBIDDEN_SQL):
        return "Помилка: агент не може змінювати базу даних."

    # Виконання запиту та отримання результату
    with engine.connect() as conn:
        result = conn.execute(text(sql_query_clean))
        rows = result.fetchall()
        columns = result.keys()

    if not rows:
        return "На жаль, результатів за вашим запитом не знайдено."

    # Формуємо людську відповідь через LLM
    # Конвертуємо результат у список словників
    table = [dict(zip(columns, row)) for row in rows]

    # Передаємо дані назад LLM для красивої відповіді
    data_msg = HumanMessage(
        content=f"Використовуючи ці дані, створи ввічливу природну відповідь для користувача:\n{table}"
    )

    final_response = llm.generate([[data_msg]])
    return final_response.generations[0][0].text.strip()


# --- Приклад використання ---
if __name__ == "__main__":
    query = "які у нас були фінансування."
    response = run_read_only_agent(query)
    print(response)
