from pinecone import Pinecone
import os
import dotenv

# завантажуємо ключ з .env
dotenv.load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY не знайдено у .env файлі")

# підключаємося
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("hospital-docs")

# повне очищення індексу (видалимо всі вектори)
index.delete(delete_all=True)

print("Index 'hospital-docs' is fully cleaned")