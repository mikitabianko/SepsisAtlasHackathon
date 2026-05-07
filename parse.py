import os
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader

def load_medical_pdfs(data_dir="./articles"):
    # Проверяем, существует ли папка
    if not os.path.exists(data_dir):
        print(f"❌ Ошибка: Папка '{data_dir}' не найдена. Создайте её и положите туда PDF.")
        return []

    # 1. Инициализируем парсер PyMuPDF. Он отлично справляется с многоколоночным текстом.
    parser = PyMuPDFReader()

    # 2. Указываем LlamaIndex использовать этот парсер для всех файлов с расширением .pdf
    file_extractor = {".pdf": parser}

    print(f"⏳ Начинаем чтение документов из папки '{data_dir}'...")
    
    # 3. Загружаем данные
    documents = SimpleDirectoryReader(
        input_dir=data_dir,
        file_extractor=file_extractor,
        required_exts=[".pdf"] # На всякий случай игнорируем всё, кроме PDF
    ).load_data()

    print(f"✅ Успешно загружено! Всего извлечено сущностей (страниц): {len(documents)}\n")
    return documents

# --- Тестируем работу скрипта ---
if __name__ == "__main__":
    docs = load_medical_pdfs("./articles") # Указываем вашу папку
    
    if docs:
        first_page = docs[0]
        print("--- Все доступные метаданные ---")
        print(first_page.metadata) # Выведем всё, чтобы найти нужный ключ
        
        # Пытаемся достать страницу разными способами
        page_num = first_page.metadata.get('page_label') or first_page.metadata.get('page') or first_page.metadata.get('page_num', 'Не найдено')
        
        print("\n--- Проверка ---")
        print(f"📄 Файл: {first_page.metadata.get('file_name')}")
        print(f"🔖 Страница: {page_num}")