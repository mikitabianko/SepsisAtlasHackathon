import os
import json
from pydantic import BaseModel, Field
import pandas as pd
from collections import defaultdict

from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from parse import load_medical_pdfs # Ваш парсер из шага 1

# --- Настройки ---
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

OPENROUTER_API_KEY = "" 

Settings.llm = OpenAILike(
    model="openai/gpt-4o-mini",
    api_key=OPENROUTER_API_KEY,
    api_base="https://openrouter.ai/api/v1",
    is_chat_model=True,
    context_window=128000,
    default_headers={"HTTP-Referer": "https://github.com", "X-Title": "Sepsis Hackathon"}
)

# --- Схема данных (добавим название статьи) ---
class SepsisExtraction(BaseModel):
    study_name: str = Field(..., description="Name of the file or study")
    population: str = Field(..., description="Cohort description, e.g., 'ED patients with suspected infection'")
    sample_size: str = Field(..., description="Total N, died N, survived N. E.g., 'N=241; Died N=41'")
    predictor: str = Field(..., description="Prognostic biomarkers or variables (e.g., lactate, lymphocytes)")
    outcome: str = Field(..., description="Outcome definition (e.g., 28-day mortality)")
    timing: str = Field(..., description="Measurement timing. E.g., 'Within first 6h of admission'")
    method: str = Field(..., description="Statistical methods used. E.g., 'Logistic regression'")
    effect_size: str = Field(..., description="Effect size (OR, HR, Cutoff). If not found, strictly write 'Not reported'")
    performance: str = Field(..., description="AUC, Sensitivity, Specificity, 95% CI. If not found, write 'Not reported'")
    source_anchor: str = Field(..., description="Exact short quote from the text proving the results")
    

# --- Главная функция сборки ---
def build_sepsis_atlas():
    print("📥 Загрузка всех документов...")
    all_documents = load_medical_pdfs("articles/articles")
    
    # Группируем страницы по имени файла
    docs_by_file = defaultdict(list)
    for doc in all_documents:
        file_name = doc.metadata.get('file_name', 'Unknown_study')
        docs_by_file[file_name].append(doc)
        
    print(f"📊 Найдено уникальных статей: {len(docs_by_file)}")
    
    atlas_data = [] # Здесь будем хранить строки нашей таблицы
    
    # Запускаем цикл по каждой статье
    for file_name, docs in docs_by_file.items():
        print(f"\n🔍 Анализ статьи: {file_name}...")
        
        # Создаем мини-индекс только для одной статьи (работает мгновенно)
        index = VectorStoreIndex.from_documents(docs)
        query_engine = index.as_query_engine(output_cls=SepsisExtraction, response_mode="compact")
        
        query = f"Extract counterfactual mortality estimation data from this study according to the schema. Focus on cohort characteristics (population, sample size), predictors, outcomes, and statistical performance (AUC, OR). Set study_name to '{file_name}'. If any specific field is missing, strictly write 'Not reported'."
        
        response = query_engine.query(query)
        
        # Достаем данные (используем наш надежный метод)
        extracted_dict = None
        if isinstance(response.response, BaseModel):
            extracted_dict = response.response.model_dump()
        elif hasattr(response, 'obj') and response.obj is not None:
            extracted_dict = response.obj.model_dump()
            
        if extracted_dict:
            atlas_data.append(extracted_dict)
            print(f"✅ Данные извлечены: {extracted_dict['predictor']} -> {extracted_dict['outcome']}")
        else:
            print(f"⚠️ Не удалось извлечь структурированные данные для {file_name}")

    # --- Превращаем результаты в таблицу Pandas ---
    if atlas_data:
        print("\n🛠 Формирование итогового Sepsis Atlas...")
        df = pd.DataFrame(atlas_data)
        
        # Сохраняем в CSV
        df.to_csv("sepsis_atlas_results.csv", index=False)
        print("💾 Таблица сохранена в 'sepsis_atlas_results.csv'!")
        
        # Выводим в консоль красивую табличку
        print("\n" + "="*50)
        print(df[['study_name', 'predictor', 'outcome', 'effect_size']].to_markdown())
        print("="*50)
    else:
        print("\n❌ Не удалось собрать данные ни из одной статьи.")

if __name__ == "__main__":
    build_sepsis_atlas()