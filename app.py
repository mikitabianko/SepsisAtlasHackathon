import streamlit as st
import pandas as pd

# Настройка страницы
st.set_page_config(page_title="Sepsis Atlas AI", layout="wide")

st.title("🧬 Sepsis Atlas: Paper-to-Knowledge Pipeline")
st.markdown("""
Этот прототип автоматически извлекает предикторы смертности, исходы и размеры эффектов (AUC, OR) из неструктурированных научных статей (PDF) с помощью LLM, сохраняя строгую привязку к источнику.
""")

# Загрузка наших данных
try:
    df = pd.read_csv("sepsis_atlas_results.csv")
    
    # Статистика
    st.success(f"✅ Успешно проанализировано статей: {len(df)}")
    
    # Интерактивная таблица
    st.dataframe(
        df,
        column_config={
            "study_name": "Исследование (Источник)",
            "predictor": "Предиктор (Биомаркер)",
            "outcome": "Исход",
            "effect_size": "Размер эффекта (AUC/OR)",
            "method": "Метод анализа",
            "source_anchor": "Цитата-подтверждение"
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Кнопка для скачивания (требование "analysis-ready data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Скачать базу данных (CSV)",
        data=csv,
        file_name='sepsis_atlas_export.csv',
        mime='text/csv',
    )

except FileNotFoundError:
    st.error("Файл sepsis_atlas_results.csv не найден. Пожалуйста, запустите скрипт извлечения данных (build_atlas.py) сначала.")