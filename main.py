import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="Dashboard ENEM 2024",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Dashboard ENEM 2024")
st.markdown("Análise exploratória dos dados de participantes do ENEM 2024")

@st.cache_resource
def get_engine():
    db_config = st.secrets["database"]
    connection_string = (
        f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    return create_engine(connection_string)

@st.cache_data
def load_data():
    engine = get_engine()
    df = pd.read_sql("SELECT q021 FROM public.ed_enem_2024_participantes limit 50", engine)
    return df

df = load_data()

st.subheader("📊 Análise da Coluna: q021 (TV a Cabo)")

freq_tv = df['q021'].value_counts().reset_index()
freq_tv.columns = ['Categoria', 'Frequência']

col1, col2 = st.columns(2)

with col1:
    fig_bar = px.bar(freq_tv, x='Categoria', y='Frequência', 
                 title='Distribuição de Frequência - TV a Cabo',
                 labels={'Categoria': 'Resposta', 'Frequência': 'Quantidade'},
                 color='Frequência',
                 color_continuous_scale='Viridis')
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.dataframe(freq_tv, use_container_width=True)
    
    st.metric("Total de Respondentes", len(df))
    
    sim_count = freq_tv[freq_tv['Categoria'].str.contains('Sim', na=False)]['Frequência'].sum()
    percentual_sim = (sim_count / len(df) * 100)
    st.metric("Percentual com TV a Cabo", f"{percentual_sim:.1f}%")

st.markdown("---")
st.caption("Projeto ENEM 2024 • Streamlit + PostgreSQL + Plotly")
