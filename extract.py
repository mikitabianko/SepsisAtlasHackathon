import os
import json
import re
from pydantic import BaseModel, Field
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai_like import OpenAILike
from parse import load_medical_pdfs # Ваш парсер
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import BaseModel


# --- НОВОЕ: Настройка локальных эмбеддингов ---
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 1. Настройка OpenRouter (маскируемся под OpenAI клиент)
OPENROUTER_API_KEY = "" # Вставьте ваш ключ сюда (лучше использовать os.environ.get)

llm = OpenAILike(
    model="openai/gpt-4o-mini",
    api_key=OPENROUTER_API_KEY,
    api_base="https://openrouter.ai/api/v1",
    is_chat_model=True,
    context_window=128000, # Явно указываем контекст для gpt-4o-mini
    default_headers={
        "HTTP-Referer": "https://github.com",
        "X-Title": "Sepsis Hackathon"
    }
)
Settings.llm = llm

# 2. Схема данных (Ваша итоговая таблица)
class SepsisExtraction(BaseModel):
    predictor: str = Field(..., description="Prognostic biomarkers or variables (e.g., lactate, lymphocytes)")
    outcome: str = Field(..., description="Outcome definition (e.g., 28-day mortality)")
    effect_size: str = Field(..., description="Effect size like AUC, OR, HR with confidence intervals. If not found, write 'Not reported'")
    method: str = Field(..., description="Statistical methods used")
    source_anchor: str = Field(..., description="Exact short quote from the text proving the effect size")

# 3. Основной пайплайн
def extract_data():
    print("Загрузка документов...")
    documents = load_medical_pdfs("./articles")
    
    print("Создание векторного индекса (в памяти)...")
    # Теперь это сработает, так как используются бесплатные локальные эмбеддинги
    index = VectorStoreIndex.from_documents(documents)
    
    print("Настройка движка запросов...")
    query_engine = index.as_query_engine(
        output_cls=SepsisExtraction,
        response_mode="compact"
    )
    
    query = "Find information about counterfactual mortality estimation, specifically predictors, outcomes, and effect sizes (AUC, OR) for sepsis patients."
    
    print("Отправка запроса в LLM...\n")
    response = query_engine.query(query)
    
    # Делаем запрос более жестким для LLM
    query = """
    Find information about counterfactual mortality estimation, specifically predictors, outcomes, and effect sizes (AUC, OR) for sepsis patients.
    IMPORTANT: You must respond ONLY with a valid JSON object matching the requested schema. Do not include markdown formatting or any other text.
    """
    
    print("Отправка запроса в LLM...\n")
    response = query_engine.query(query)
    
    print("Отправка запроса в LLM...\n")
    response = query_engine.query(query)
    
    # 1. Если LlamaIndex положил готовый Pydantic-объект прямо в response.response (Ваш случай!)
    if isinstance(response.response, BaseModel):
        print("🎉 Успех! Получен структурированный объект:\n")
        print(response.response.model_dump_json(indent=2))
        
    # 2. Если объект лежит в response.obj (Старое поведение)
    elif hasattr(response, 'obj') and response.obj is not None:
        print("🎉 Успех! Получен структурированный объект:\n")
        print(response.obj.model_dump_json(indent=2))
        
    # 3. Если вернулся просто текст (Фолбэк)
    elif isinstance(response.response, str):
        print("⚠️ Вернулся текст. Достаем JSON вручную...")
        raw_text = response.response
        import re, json
        try:
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                parsed_dict = json.loads(match.group(0))
                print(json.dumps(parsed_dict, indent=2, ensure_ascii=False))
            else:
                print("❌ JSON не найден. Текст:", raw_text)
        except json.JSONDecodeError:
            print("❌ Ошибка парсинга JSON. Текст:", raw_text)
    else:
        print("Неизвестный формат ответа:", type(response.response))

if __name__ == "__main__":
    extract_data()
    
