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

engine = get_engine()
df = pd.read_sql("SELECT q003,q004 FROM public.ed_enem_2024_participantes", engine)

st.subheader("📊 Análise da Coluna: Posição Social")

freq_posicao_pai = df['q003'].value_counts().reset_index()
freq_posicao_pai.columns = ['Categoria', 'Frequência']

freq_posicao_mae = df['q004'].value_counts().reset_index()
freq_posicao_mae.columns = ['Categoria', 'Frequência']

col1, col2 = st.columns(2)

with col1:
    fig_bar_pai = px.bar(
        freq_posicao_pai,
        x='Categoria',
        y='Frequência',
        title='Distribuição de Frequência - Posição Social (Pai)',
        labels={'Categoria': 'Resposta', 'Frequência': 'Quantidade'},
        color='Frequência',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_bar_pai, use_container_width=True)

with col2:
    st.dataframe(freq_posicao_pai, use_container_width=True)
    st.metric("Total de Respondentes", len(df))

freq_posicao_pai['Grupo'] = freq_posicao_pai['Categoria'].str.extract(r'Grupo (\d+)').astype(float)
freq_posicao_mae['Grupo'] = freq_posicao_mae['Categoria'].str.extract(r'Grupo (\d+)').astype(float)

freq_pai = freq_posicao_pai.groupby('Grupo')['Frequência'].sum().reset_index()
freq_mae = freq_posicao_mae.groupby('Grupo')['Frequência'].sum().reset_index()

todos_grupos = pd.DataFrame({
    'Grupo': sorted(set(freq_pai['Grupo'].dropna()).union(set(freq_mae['Grupo'].dropna())))
})

freq_pai = pd.merge(todos_grupos, freq_pai, on='Grupo', how='left').fillna(0)
freq_mae = pd.merge(todos_grupos, freq_mae, on='Grupo', how='left').fillna(0)

df_radar = pd.DataFrame({
    'Grupo': todos_grupos['Grupo'].astype(int).astype(str).apply(lambda x: f'Grupo {x}'),
    'Pai': freq_pai['Frequência'],
    'Mãe': freq_mae['Frequência']
})

fig_radar = px.line_polar(
    df_radar,
    r='Pai',
    theta='Grupo',
    line_close=True
)

fig_radar.add_scatterpolar(
    r=df_radar['Pai'],
    theta=df_radar['Grupo'],
    fill='toself',
    name='Pai'
)

fig_radar.add_scatterpolar(
    r=df_radar['Mãe'],
    theta=df_radar['Grupo'],
    fill='toself',
    name='Mãe'
)

fig_radar.update_layout(
    template='plotly_dark',
    title='Comparação por Grupo - Pai vs Mãe',
    title_x=0.5,
    polar=dict(
        radialaxis=dict(visible=True)
    )
)

st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")
st.caption("Projeto ENEM 2024 • Streamlit + PostgreSQL + Plotly")
