import streamlit as st
import pandas as pd
import json

model_prediction_file = open("model_prediction.json", "r")
data = json.load(model_prediction_file)

df = pd.DataFrame(data)

st.title("Sepsis Evidence Table")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "paper_id": "Paper ID",
        "title": st.column_config.TextColumn(
            "Title",
            width="medium"
        ),
        "sample_size": st.column_config.NumberColumn(
            "N",
            format="%d"
        ),
        "p_value": st.column_config.TextColumn(
            "P-value"
        ),
        "source_quote": st.column_config.TextColumn(
            "Evidence Quote",
            width="large"
        ),
    }
)