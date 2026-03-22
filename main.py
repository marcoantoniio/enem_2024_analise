import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import plotly.express as px

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

query = """
WITH pai AS (
    SELECT 
        CASE 
            WHEN q003 LIKE 'Grupo %' THEN SUBSTRING(q003 FROM 'Grupo [0-9]+')
            ELSE 'Não sei'
        END AS grupo,
        COUNT(*) AS freq_pai
    FROM public.ed_enem_2024_participantes
    GROUP BY grupo
),
mae AS (
    SELECT 
        CASE 
            WHEN q004 LIKE 'Grupo %' THEN SUBSTRING(q004 FROM 'Grupo [0-9]+')
            ELSE 'Não sei'
        END AS grupo,
        COUNT(*) AS freq_mae
    FROM public.ed_enem_2024_participantes
    GROUP BY grupo
)
SELECT 
    COALESCE(pai.grupo, mae.grupo) AS grupo,
    COALESCE(freq_pai, 0) AS pai,
    COALESCE(freq_mae, 0) AS mae
FROM pai
FULL OUTER JOIN mae ON pai.grupo = mae.grupo
ORDER BY grupo;
"""

df = pd.read_sql(text(query), engine)

df = df[df['grupo'] != 'Não sei']

df['pai'] = df['pai'] / df['pai'].sum()
df['mae'] = df['mae'] / df['mae'].sum()

fig_radar = px.line_polar(
    df,
    r='pai',
    theta='grupo',
    line_close=True
)

fig_radar.add_scatterpolar(
    r=df['pai'],
    theta=df['grupo'],
    fill='toself',
    name='Pai'
)

fig_radar.add_scatterpolar(
    r=df['mae'],
    theta=df['grupo'],
    fill='toself',
    name='Mãe'
)

fig_radar.update_layout(
    template='plotly_dark',
    title='Comparação (%) por Grupo - Pai vs Mãe',
    title_x=0.5,
    polar=dict(
        radialaxis=dict(visible=True, tickformat=".0%")
    )
)

st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")
st.caption("Projeto ENEM 2024 • Streamlit + PostgreSQL + Plotly")
