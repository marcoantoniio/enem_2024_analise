import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import math
import random
from sqlalchemy import create_engine, text
from scipy.stats import shapiro, norm

st.set_page_config(page_title="Dashboard ENEM - Amostragem", layout="wide")

@st.cache_resource
def get_engine():
    db_config = st.secrets["database"]
    return create_engine(
        f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )

engine = get_engine()

if 'total_participantes' not in st.session_state:
    with engine.connect() as conn:
        st.session_state.total_participantes = conn.execute(text("SELECT COUNT(*) FROM public.ed_enem_2024_participantes")).scalar()
        st.session_state.total_resultados = conn.execute(text("SELECT COUNT(*) FROM public.ed_enem_2024_resultados")).scalar()

if 'cache_amostras' not in st.session_state:
    st.session_state.cache_amostras = {}

st.sidebar.image("logo_enem.png", width=350)
st.sidebar.title("Configuração de Dados")
st.sidebar.markdown("Altere a base de dados para recalcular todo o painel automaticamente.")

fonte_dados = st.sidebar.radio(
    "Selecione a Fonte de Análise:",
    ["População Total (Real-time)",
     "Amostra Aleatória Simples",
     "Amostra Sistemática",
     "Amostra Estratificada"]
)

n_amostra = 9604

st.sidebar.divider()
st.sidebar.markdown("### Dados da Amostra")
st.sidebar.write(f"**Nível de Confiança:** 95%\n\n**Margem de Erro:** 1%\n\n**Tamanho:** {n_amostra:,}".replace(',', '.'))

if "População" in fonte_dados:
    st.session_state.last_extracted_fonte = None 
elif fonte_dados not in st.session_state.cache_amostras:
    with st.spinner(f"Extraindo {fonte_dados} (tamanho = {n_amostra})"):
        total_p = st.session_state.total_participantes
        total_r = st.session_state.total_resultados
        
        if fonte_dados == "Amostra Aleatória Simples":
            q_p = f"SELECT * FROM public.ed_enem_2024_participantes TABLESAMPLE SYSTEM(5) LIMIT {n_amostra}"
            q_r = f"SELECT * FROM public.ed_enem_2024_resultados TABLESAMPLE SYSTEM(5) LIMIT {n_amostra}"
            
        elif fonte_dados == "Amostra Sistemática":
            k_p = total_p // n_amostra
            start_p = random.randint(1, k_p)
            q_p = f"SELECT * FROM public.ed_enem_2024_participantes WHERE CAST(nu_inscricao AS BIGINT) % {k_p} = {start_p} LIMIT {n_amostra}"
            
            k_r = total_r // n_amostra
            start_r = random.randint(1, k_r)
            q_r = f"SELECT * FROM public.ed_enem_2024_resultados WHERE CAST(nu_sequencial AS BIGINT) % {k_r} = {start_r} LIMIT {n_amostra}"
            
        elif fonte_dados == "Amostra Estratificada":
            q_p = f"""
                WITH amostra_bruta AS (
                    SELECT * FROM public.ed_enem_2024_participantes TABLESAMPLE SYSTEM(10)
                ),
                pop_counts AS (
                    SELECT regiao_nome_prova, COUNT(*) as total_estrato
                    FROM amostra_bruta WHERE regiao_nome_prova IS NOT NULL GROUP BY regiao_nome_prova
                ),
                ranked AS (
                    SELECT p.*, ROW_NUMBER() OVER(PARTITION BY p.regiao_nome_prova) as rn, c.total_estrato
                    FROM amostra_bruta p JOIN pop_counts c ON p.regiao_nome_prova = c.regiao_nome_prova
                    WHERE p.regiao_nome_prova IS NOT NULL
                )
                SELECT * FROM ranked WHERE rn <= ROUND({n_amostra} * (total_estrato::numeric / (SELECT COUNT(*) FROM amostra_bruta)))
            """
            q_r = f"SELECT * FROM public.ed_enem_2024_resultados TABLESAMPLE SYSTEM(5) LIMIT {n_amostra}"

        with engine.connect() as conn:
            df_p = pd.read_sql(text(q_p), conn)
            df_r = pd.read_sql(text(q_r), conn)
            
            st.session_state.cache_amostras[fonte_dados] = {'participantes': df_p, 'resultados': df_r}

if "População" not in fonte_dados:
    st.session_state.df_participantes_amostra = st.session_state.cache_amostras[fonte_dados]['participantes']
    st.session_state.df_resultados_amostra = st.session_state.cache_amostras[fonte_dados]['resultados']
    st.session_state.last_extracted_fonte = fonte_dados

st.title(f"📊 Indicadores Principais - {fonte_dados}")


@st.cache_data(show_spinner="Buscando opções...")
def get_filter_options():
    with engine.connect() as conn:
        racas = pd.read_sql(text("SELECT DISTINCT tp_cor_raca FROM public.ed_enem_2024_participantes WHERE tp_cor_raca IS NOT NULL AND tp_cor_raca NOT ILIKE '%issing%'"), conn)["tp_cor_raca"].tolist()
        regioes = pd.read_sql(text("SELECT DISTINCT regiao_nome_prova FROM public.ed_enem_2024_participantes WHERE regiao_nome_prova IS NOT NULL AND regiao_nome_prova NOT ILIKE '%issing%'"), conn)["regiao_nome_prova"].tolist()
        banheiros = pd.read_sql(text("SELECT DISTINCT q009 FROM public.ed_enem_2024_participantes WHERE q009 IS NOT NULL AND q009 NOT ILIKE '%issing%'"), conn)["q009"].tolist()
    return racas, regioes, banheiros

@st.cache_data(show_spinner=False)
def get_nacionalidade_data(fonte):
    if "População" in fonte:
        query = "SELECT tp_nacionalidade, COUNT(*) AS frequencia FROM public.ed_enem_2024_participantes GROUP BY tp_nacionalidade;"
        with engine.connect() as conn:
            return pd.read_sql(text(query), conn)
    else:
        df = st.session_state.df_participantes_amostra
        return df.groupby('tp_nacionalidade').size().reset_index(name='frequencia')

@st.cache_data(show_spinner=False)
def get_dashboard_data(column_name, racas_selecionadas=(), regioes_selecionadas=(), banheiros_selecionados=(), fonte="População"):
    if "População" in fonte:
        where_clauses = ["1=1"]
        where_clauses.append(f"CAST({column_name} AS TEXT) NOT ILIKE '%issing%'")
        if racas_selecionadas: where_clauses.append(f"tp_cor_raca IN ('{'', ''.join(racas_selecionadas)}')")
        if regioes_selecionadas: where_clauses.append(f"regiao_nome_prova IN ('{'', ''.join(regioes_selecionadas)}')")
        if banheiros_selecionados: where_clauses.append(f"q009 IN ('{'', ''.join(banheiros_selecionados)}')")
        query = f"SELECT {column_name} AS categoria, COUNT(*) AS frequencia FROM public.ed_enem_2024_participantes WHERE {column_name} IS NOT NULL AND {' AND '.join(where_clauses)} GROUP BY {column_name};"
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        if not df.empty: df["percentual"] = df["frequencia"] * 100.0 / df["frequencia"].sum()
        return df
    else:
        df = st.session_state.df_participantes_amostra
        df = df[~df[column_name].astype(str).str.contains('issing', case=False, na=False)]
        if racas_selecionadas: df = df[df['tp_cor_raca'].isin(racas_selecionadas)]
        if regioes_selecionadas: df = df[df['regiao_nome_prova'].isin(regioes_selecionadas)]
        if banheiros_selecionados: df = df[df['q009'].isin(banheiros_selecionados)]
        res = df.groupby(column_name).size().reset_index(name='frequencia')
        res.rename(columns={column_name: 'categoria'}, inplace=True)
        res['percentual'] = res['frequencia'] * 100.0 / res['frequencia'].sum() if res['frequencia'].sum() > 0 else 0
        return res

@st.cache_data(show_spinner=False)
def get_bens_consolidados(racas_selecionadas=(), regioes_selecionadas=(), fonte="População"):
    if "População" in fonte:
        where_clauses = ["1=1"]
        if racas_selecionadas: where_clauses.append(f"tp_cor_raca IN ('{'', ''.join(racas_selecionadas)}')")
        if regioes_selecionadas: where_clauses.append(f"regiao_nome_prova IN ('{'', ''.join(regioes_selecionadas)}')")
        query = f"""
            SELECT COUNT(*) as total,
            SUM(CASE WHEN q016 = 'Sim' THEN 1 ELSE 0 END) as micro_sim, SUM(CASE WHEN q016 = 'Não' THEN 1 ELSE 0 END) as micro_nao,
            SUM(CASE WHEN q020 = 'Sim' THEN 1 ELSE 0 END) as wifi_sim, SUM(CASE WHEN q020 = 'Não' THEN 1 ELSE 0 END) as wifi_nao,
            SUM(CASE WHEN q022 = 'Não' THEN 1 ELSE 0 END) as cel_nao, SUM(CASE WHEN q022 = 'Sim, um' THEN 1 ELSE 0 END) as cel_1,
            SUM(CASE WHEN q022 = 'Sim, dois' THEN 1 ELSE 0 END) as cel_2, SUM(CASE WHEN q022 = 'Sim, três' THEN 1 ELSE 0 END) as cel_3,
            SUM(CASE WHEN q022 = 'Sim, quatro ou mais' THEN 1 ELSE 0 END) as cel_4, SUM(CASE WHEN q021 = 'Não' THEN 1 ELSE 0 END) as pc_nao,
            SUM(CASE WHEN q021 = 'Sim, um' THEN 1 ELSE 0 END) as pc_1, SUM(CASE WHEN q021 = 'Sim, dois' THEN 1 ELSE 0 END) as pc_2,
            SUM(CASE WHEN q021 = 'Sim, três' THEN 1 ELSE 0 END) as pc_3, SUM(CASE WHEN q021 = 'Sim, quatro ou mais' THEN 1 ELSE 0 END) as pc_4
            FROM public.ed_enem_2024_participantes WHERE {' AND '.join(where_clauses)};
        """
        with engine.connect() as conn:
            return pd.read_sql(text(query), conn)
    else:
        df = st.session_state.df_participantes_amostra
        if racas_selecionadas: df = df[df['tp_cor_raca'].isin(racas_selecionadas)]
        if regioes_selecionadas: df = df[df['regiao_nome_prova'].isin(regioes_selecionadas)]
        return pd.DataFrame([{
            'total': len(df),
            'micro_sim': (df['q016'] == 'Sim').sum(), 'micro_nao': (df['q016'] == 'Não').sum(),
            'wifi_sim': (df['q020'] == 'Sim').sum(), 'wifi_nao': (df['q020'] == 'Não').sum(),
            'cel_nao': (df['q022'] == 'Não').sum(), 'cel_1': (df['q022'] == 'Sim, um').sum(),
            'cel_2': (df['q022'] == 'Sim, dois').sum(), 'cel_3': (df['q022'] == 'Sim, três').sum(),
            'cel_4': (df['q022'] == 'Sim, quatro ou mais').sum(), 'pc_nao': (df['q021'] == 'Não').sum(),
            'pc_1': (df['q021'] == 'Sim, um').sum(), 'pc_2': (df['q021'] == 'Sim, dois').sum(),
            'pc_3': (df['q021'] == 'Sim, três').sum(), 'pc_4': (df['q021'] == 'Sim, quatro ou mais').sum()
        }])

@st.cache_data(show_spinner=False)
def get_escolaridade_pais(racas_selecionadas=(), regioes_selecionadas=(), fonte="População"):
    if "População" in fonte:
        where_clauses = ["1=1"]
        if racas_selecionadas: where_clauses.append(f"tp_cor_raca IN ('{'', ''.join(racas_selecionadas)}')")
        if regioes_selecionadas: where_clauses.append(f"regiao_nome_prova IN ('{'', ''.join(regioes_selecionadas)}')")
        query = f"""
            SELECT 'pai' AS responsavel, q001 AS categoria, COUNT(*) AS frequencia FROM public.ed_enem_2024_participantes WHERE q001 IS NOT NULL AND CAST(q001 AS TEXT) NOT ILIKE '%issing%' AND {' AND '.join(where_clauses)} GROUP BY q001
            UNION ALL
            SELECT 'mae' AS responsavel, q002 AS categoria, COUNT(*) AS frequencia FROM public.ed_enem_2024_participantes WHERE q002 IS NOT NULL AND CAST(q002 AS TEXT) NOT ILIKE '%issing%' AND {' AND '.join(where_clauses)} GROUP BY q002;
        """
        with engine.connect() as conn: return pd.read_sql(text(query), conn)
    else:
        df = st.session_state.df_participantes_amostra
        if racas_selecionadas: df = df[df['tp_cor_raca'].isin(racas_selecionadas)]
        if regioes_selecionadas: df = df[df['regiao_nome_prova'].isin(regioes_selecionadas)]
        pai = df.groupby('q001').size().reset_index(name='frequencia').rename(columns={'q001': 'categoria'})
        pai['responsavel'] = 'pai'
        mae = df.groupby('q002').size().reset_index(name='frequencia').rename(columns={'q002': 'categoria'})
        mae['responsavel'] = 'mae'
        res = pd.concat([pai, mae], ignore_index=True)
        return res[~res['categoria'].astype(str).str.contains('issing', case=False, na=False)]

@st.cache_data(show_spinner=False)
def get_amostra_notas(fonte="População"):
    if "População" in fonte:
        query = """SELECT nota_cn_ciencias_da_natureza AS "C. da Natureza", nota_ch_ciencias_humanas AS "Ciências Humanas", nota_lc_linguagens_e_codigos AS "Linguagens", nota_mt_matematica AS "Matemática", nota_redacao AS "Redação", nota_media_5_notas AS "Média Geral" FROM public.ed_enem_2024_resultados LIMIT 30000;"""
        with engine.connect() as conn: return pd.read_sql(text(query), conn)
    else:
        df = st.session_state.df_resultados_amostra
        res = df[['nota_cn_ciencias_da_natureza', 'nota_ch_ciencias_humanas', 'nota_lc_linguagens_e_codigos', 'nota_mt_matematica', 'nota_redacao', 'nota_media_5_notas']].copy()
        res.columns = ["C. da Natureza", "Ciências Humanas", "Linguagens", "Matemática", "Redação", "Média Geral"]
        return res

def format_kpi_value(val):
    if val == 0: return "0%"
    if val < 5: return f"{val:.1f}%"
    return f"{val:.0f}%"

def get_percentage(df, category_col, category_name):
    if df.empty: return 0.0
    try: return float(df.loc[df[category_col] == category_name, "percentual"].iloc[0])
    except IndexError: return 0.0

def render_kpi_html(valor, label, font_size="35px", label_size="14px"):
    return f"<div style='text-align: center; background-color: transparent;'><div style='color: #268C82; font-size: {font_size}; font-weight: bold; line-height: 1.1;'>{format_kpi_value(valor)}</div><div style='color: #888; font-size: {label_size};'>{label}</div></div>"


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Nacionalidade", "Dashboard (Cor, Região, Bens)", "Escolaridade Pais",
    "Banheiros e Renda", "Distribuição de Notas", "Normalidade (Shapiro-Wilk)",
    "Perfil Demográfico", "Validação da Amostra"
])

racas_opcoes, regioes_opcoes, banheiros_opcoes = get_filter_options()

with tab1:
    st.subheader("Indicadores de Nacionalidade")
    df_nac = get_nacionalidade_data(fonte_dados)
    if not df_nac.empty:
        total_nac = df_nac["frequencia"].sum()
        df_nac["percentual"] = df_nac["frequencia"] * 100.0 / total_nac
        estrangeiro = get_percentage(df_nac, "tp_nacionalidade", "Estrangeiro(a)")
        nao_inf = get_percentage(df_nac, "tp_nacionalidade", "Não informado")
        outros = df_nac[~df_nac["tp_nacionalidade"].isin(["Estrangeiro(a)", "Não informado"])]["percentual"].sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Estrangeiro(a)", f"{estrangeiro:.2f}%")
        col2.metric("Não informado", f"{nao_inf:.2f}%")
        col3.metric("Outros", f"{outros:.2f}%")

with tab2:
    st.write("") 
    st.markdown("##### Filtros Interativos")
    col_filtro1, col_filtro2 = st.columns(2)
    with col_filtro1: f_raca = st.multiselect("Filtrar por Cor e Raça:", options=racas_opcoes, default=[])
    with col_filtro2: f_regiao = st.multiselect("Filtrar por Região:", options=regioes_opcoes, default=[])
    st.divider()

    col_esquerda, col_direita = st.columns([1, 1], gap="large")

    with col_esquerda:
        st.markdown("<h6 style='color: #888;'>Frequência por cor e raça</h6>", unsafe_allow_html=True)
        df_raca = get_dashboard_data("tp_cor_raca", (), tuple(f_regiao), (), fonte_dados)
        if not df_raca.empty:
            fig_raca = px.pie(df_raca, values='frequencia', names='categoria', hole=0.6)
            fig_raca.update_traces(textposition='inside', textinfo='percent', rotation=90)
            fig_raca.update_layout(showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0), margin=dict(t=10, b=10, l=10, r=10), height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_raca, use_container_width=True)

        st.markdown("<h6 style='color: #888;'>Frequência por região</h6>", unsafe_allow_html=True)
        df_regiao_dash = get_dashboard_data("regiao_nome_prova", tuple(f_raca), (), (), fonte_dados) 
        if not df_regiao_dash.empty:
            fig_regiao = px.treemap(df_regiao_dash, path=['categoria'], values='frequencia', color_continuous_scale='GnBu')
            fig_regiao.update_layout(margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_regiao, use_container_width=True)

    with col_direita:
        df_bens = get_bens_consolidados(tuple(f_raca), tuple(f_regiao), fonte_dados)
        if not df_bens.empty and df_bens['total'].iloc[0] > 0:
            row = df_bens.iloc[0]
            total = row['total']
            
            st.markdown("<h6 style='color: #888;'>Tem micro-ondas na sua casa?</h6>", unsafe_allow_html=True)
            k1, k2 = st.columns(2)
            k1.markdown(render_kpi_html((row['micro_nao'] / total) * 100, "Não", font_size="55px"), unsafe_allow_html=True)
            k2.markdown(render_kpi_html((row['micro_sim'] / total) * 100, "Sim", font_size="55px"), unsafe_allow_html=True)
            st.write("\n\n")
            
            st.markdown("<h6 style='color: #888;'>Tem celulares na residência?</h6>", unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.markdown(render_kpi_html((row['cel_3'] / total) * 100, "Sim, três", font_size="32px", label_size="11px"), unsafe_allow_html=True)
            c2.markdown(render_kpi_html((row['cel_2'] / total) * 100, "Sim, dois", font_size="32px", label_size="11px"), unsafe_allow_html=True)
            c3.markdown(render_kpi_html((row['cel_4'] / total) * 100, "Sim, quatro ou mais", font_size="32px", label_size="10px"), unsafe_allow_html=True)
            c4.markdown(render_kpi_html((row['cel_1'] / total) * 100, "Sim, um", font_size="32px", label_size="11px"), unsafe_allow_html=True)
            c5.markdown(render_kpi_html((row['cel_nao'] / total) * 100, "Não", font_size="32px", label_size="11px"), unsafe_allow_html=True)
            st.write("\n\n")

            st.markdown("<h6 style='color: #888;'>Tem computador/laptop?</h6>", unsafe_allow_html=True)
            p1, p2, p3, p4, p5 = st.columns(5)
            p1.markdown(render_kpi_html((row['pc_nao'] / total) * 100, "Não", font_size="32px", label_size="11px"), unsafe_allow_html=True)
            p2.markdown(render_kpi_html((row['pc_1'] / total) * 100, "Sim, um", font_size="32px", label_size="11px"), unsafe_allow_html=True)
            p3.markdown(render_kpi_html((row['pc_2'] / total) * 100, "Sim, dois", font_size="32px", label_size="11px"), unsafe_allow_html=True)
            p4.markdown(render_kpi_html((row['pc_3'] / total) * 100, "Sim, três", font_size="32px", label_size="11px"), unsafe_allow_html=True)
            p5.markdown(render_kpi_html((row['pc_4'] / total) * 100, "Sim, quatro ou mais", font_size="32px", label_size="10px"), unsafe_allow_html=True)
            st.write("\n\n")

            st.markdown("<h6 style='color: #888;'>Tem wifi?</h6>", unsafe_allow_html=True)
            w1, w2 = st.columns(2)
            w1.markdown(render_kpi_html((row['wifi_sim'] / total) * 100, "Sim", font_size="60px", label_size="16px"), unsafe_allow_html=True)
            w2.markdown(render_kpi_html((row['wifi_nao'] / total) * 100, "Não", font_size="60px", label_size="16px"), unsafe_allow_html=True)

with tab3:
    st.write("")
    st.markdown("##### Filtros Interativos")
    col_filtro1_t3, col_filtro2_t3 = st.columns(2)
    with col_filtro1_t3: f_raca_t3 = st.multiselect("Filtrar por Cor e Raça:", options=racas_opcoes, default=[], key="raca_t3")
    with col_filtro2_t3: f_regiao_t3 = st.multiselect("Filtrar por Região:", options=regioes_opcoes, default=[], key="regiao_t3")
    st.divider()

    st.markdown("<h6 style='color: #888; text-align: center;'>Frequência por cor e raça</h6>", unsafe_allow_html=True)
    df_raca_t3 = get_dashboard_data("tp_cor_raca", (), tuple(f_regiao_t3), (), fonte_dados)
    if not df_raca_t3.empty:
        fig_raca_t3 = px.pie(df_raca_t3, values='frequencia', names='categoria', hole=0.6)
        fig_raca_t3.update_traces(textposition='inside', textinfo='percent', rotation=90)
        fig_raca_t3.update_layout(showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0), margin=dict(t=10, b=30, l=10, r=10), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_raca_t3, use_container_width=True)

    st.write("---")
    df_escolaridade = get_escolaridade_pais(tuple(f_raca_t3), tuple(f_regiao_t3), fonte_dados)
    col_pai, col_mae = st.columns(2, gap="large")

    with col_pai:
        st.markdown("<h6 style='color: #888;'>Nivel de Escolaridade do Pai</h6>", unsafe_allow_html=True)
        df_pai = df_escolaridade[df_escolaridade['responsavel'] == 'pai']
        if not df_pai.empty:
            df_pai = df_pai.sort_values(by='frequencia', ascending=True)
            df_pai['freq_formatada'] = df_pai['frequencia'].apply(lambda x: f"{x:,}".replace(',', '.'))
            fig_pai = px.bar(df_pai, x='frequencia', y='categoria', orientation='h', text='freq_formatada')
            fig_pai.update_traces(marker_color='#38B2A3', textposition='inside', insidetextanchor='end')
            fig_pai.update_layout(xaxis_title="", yaxis_title="", xaxis=dict(showgrid=True, showticklabels=False), margin=dict(l=10, r=10, t=10, b=10), height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pai, use_container_width=True)

    with col_mae:
        st.markdown("<h6 style='color: #888;'>Nivel de Escolaridade da Mãe</h6>", unsafe_allow_html=True)
        df_mae = df_escolaridade[df_escolaridade['responsavel'] == 'mae']
        if not df_mae.empty:
            df_mae = df_mae.sort_values(by='frequencia', ascending=True)
            df_mae['freq_formatada'] = df_mae['frequencia'].apply(lambda x: f"{x:,}".replace(',', '.'))
            fig_mae = px.bar(df_mae, x='frequencia', y='categoria', orientation='h', text='freq_formatada')
            fig_mae.update_traces(marker_color='#38B2A3', textposition='inside', insidetextanchor='end')
            fig_mae.update_layout(xaxis_title="", yaxis_title="", xaxis=dict(showgrid=True, showticklabels=False), margin=dict(l=10, r=10, t=10, b=10), height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_mae, use_container_width=True)

with tab4:
    st.write("")
    st.markdown("##### Filtros Interativos")
    f_banheiros = st.multiselect("Filtrar por Quantidade de Banheiros:", options=banheiros_opcoes, default=[], key="banheiros_t4")
    st.divider()

    st.markdown("<h6 style='color: #888; text-align: center;'>Quantos banheiros em casa</h6>", unsafe_allow_html=True)
    df_banheiros_dash = get_dashboard_data("q009", (), (), (), fonte_dados)
    if not df_banheiros_dash.empty:
        fig_banheiros = px.pie(df_banheiros_dash, values='frequencia', names='categoria', hole=0.6)
        fig_banheiros.update_traces(textposition='outside', textinfo='label+value+percent')
        fig_banheiros.update_layout(showlegend=False, margin=dict(t=80, b=80, l=50, r=50), height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_banheiros, use_container_width=True)

    st.write("---")
    st.markdown("<h6 style='color: #888;'>Tabela de Distribuição de Frequência - Renda Familiar</h6>", unsafe_allow_html=True)
    
    ordem_renda = [
        "Nenhuma renda", "Até R$ 1.412,00", "De R$ 1.412,01 até R$ 2.118,00", "De R$ 2.118,01 até R$ 2.824,00",
        "De R$ 2.824,01 até R$ 3.530,00", "De R$ 3.530,01 até R$ 4.236,00", "De R$ 4.236,01 até R$ 5.648,00",
        "De R$ 5.648,01 até R$ 7.060,00", "De R$ 7.060,01 até R$ 8.472,00", "De R$ 8.472,01 até R$ 9.884,00",
        "De R$ 9.884,01 até R$ 11.296,00", "De R$ 11.296,01 até R$ 12.708,00", "De R$ 12.708,01 até R$ 14.120,00",
        "De R$ 14.120,01 até R$ 16.944,00", "De R$ 16.944,01 até R$ 21.180,00", "De R$ 21.180,01 até R$ 28.240,00",
        "Acima de R$ 28.240,00"
    ]

    df_renda = get_dashboard_data("q007", (), (), tuple(f_banheiros), fonte_dados)
    if not df_renda.empty:
        df_renda['categoria'] = pd.Categorical(df_renda['categoria'], categories=ordem_renda, ordered=True)
        df_renda = df_renda.sort_values(by='categoria')
        
        total_renda = df_renda['frequencia'].sum()
        df_renda['Freq. Absoluta'] = df_renda['frequencia']
        df_renda['Freq. Abs. Acumulada'] = df_renda['frequencia'].cumsum()
        df_renda['Freq. Relativa (%)'] = (df_renda['frequencia'] / total_renda) * 100
        df_renda['Freq. Rel. Acumulada (%)'] = df_renda['Freq. Relativa (%)'].cumsum()
        
        df_renda_display = df_renda.copy()
        df_renda_display['Freq. Absoluta'] = df_renda_display['Freq. Absoluta'].apply(lambda x: f"{int(x):,}".replace(',', '.'))
        df_renda_display['Freq. Abs. Acumulada'] = df_renda_display['Freq. Abs. Acumulada'].apply(lambda x: f"{int(x):,}".replace(',', '.'))
        df_renda_display['Freq. Relativa (%)'] = df_renda_display['Freq. Relativa (%)'].apply(lambda x: f"{x:.2f}%")
        df_renda_display['Freq. Rel. Acumulada (%)'] = df_renda_display['Freq. Rel. Acumulada (%)'].apply(lambda x: f"{x:.2f}%")
        df_renda_display = df_renda_display.rename(columns={'categoria': 'Renda Familiar'})
        
        st.dataframe(df_renda_display[['Renda Familiar', 'Freq. Absoluta', 'Freq. Abs. Acumulada', 'Freq. Relativa (%)', 'Freq. Rel. Acumulada (%)']], use_container_width=True, hide_index=True)

with tab5:
    st.write("")
    st.markdown("##### Comparativo Geral de Notas")
    st.divider()
    df_amostra = get_amostra_notas(fonte_dados)
    if not df_amostra.empty:
        df_melted = df_amostra.melt(var_name="Prova", value_name="Nota").dropna()
        fig_violino = px.violin(df_melted, y="Nota", x="Prova", color="Prova", box=True, points=False)
        fig_violino.update_layout(showlegend=False, xaxis_title="", yaxis_title="Pontuação", margin=dict(t=30, b=30, l=10, r=10), height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_violino, use_container_width=True)

with tab6:
    st.write("")
    st.markdown("##### Teste de Normalidade Estatística")
    st.divider()
    df_amostra = get_amostra_notas(fonte_dados)
    if not df_amostra.empty:
        df_teste = df_amostra[['Matemática', 'Linguagens']].dropna()
        if len(df_teste) > 5000: df_teste = df_teste.sample(n=5000, random_state=42)

        col_mt, col_lc = st.columns(2, gap="large")
        num_bins = 50
        raca_cor = '#38B2A3'
        lingua_cor = '#5C78D8'

        with col_mt:
            st.markdown("<h6 style='color: #888; text-align: center;'>Distribuição - Matemática</h6>", unsafe_allow_html=True)
            x = df_teste['Matemática']
            mu_mt, std_mt = x.mean(), x.std()
            fig_mt = go.Figure()
            fig_mt.add_trace(go.Histogram(x=x, nbinsx=num_bins, name='Histograma', marker_color=raca_cor, opacity=0.75))
            x_theoretical = np.linspace(x.min(), x.max(), 200)
            y_theoretical = norm.pdf(x_theoretical, mu_mt, std_mt)
            fator_escala = len(x) * ((x.max() - x.min()) / num_bins)
            fig_mt.add_trace(go.Scatter(x=x_theoretical, y=y_theoretical * fator_escala, mode='lines', name='Curva Normal', line=dict(color='black', width=2)))
            fig_mt.update_layout(xaxis_title="Nota", yaxis_title="Frequência", margin=dict(t=10, b=10, l=10, r=10), height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, bargap=0.01)
            st.plotly_chart(fig_mt, use_container_width=True)

            stat_mt, p_mt = shapiro(x)
            p_formatado_mt = f"{p_mt:.4f}" if p_mt >= 0.0001 else "< 0,0001"
            st.markdown(f"<div style='background-color: rgba(0,0,0,0.03); padding: 15px; border-radius: 8px; text-align: center; border: 1px solid rgba(0,0,0,0.1);'><p style='margin-bottom: 5px; color: #555; font-size: 16px;'><b>Estatística W:</b> {stat_mt:.4f}</p><p style='margin-bottom: 0px; color: #555; font-size: 16px;'><b>Valor-p:</b> {p_formatado_mt}</p></div>", unsafe_allow_html=True)

        with col_lc:
            st.markdown("<h6 style='color: #888; text-align: center;'>Distribuição - Linguagens</h6>", unsafe_allow_html=True)
            x = df_teste['Linguagens']
            mu_lc, std_lc = x.mean(), x.std()
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Histogram(x=x, nbinsx=num_bins, name='Histograma', marker_color=lingua_cor, opacity=0.75))
            x_theoretical = np.linspace(x.min(), x.max(), 200)
            y_theoretical = norm.pdf(x_theoretical, mu_lc, std_lc)
            fator_escala = len(x) * ((x.max() - x.min()) / num_bins)
            fig_lc.add_trace(go.Scatter(x=x_theoretical, y=y_theoretical * fator_escala, mode='lines', name='Curva Normal', line=dict(color='black', width=2)))
            fig_lc.update_layout(xaxis_title="Nota", yaxis_title="Frequência", margin=dict(t=10, b=10, l=10, r=10), height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, bargap=0.01)
            st.plotly_chart(fig_lc, use_container_width=True)

            stat_lc, p_lc = shapiro(x)
            p_formatado_lc = f"{p_lc:.4f}" if p_lc >= 0.0001 else "< 0,0001"
            st.markdown(f"<div style='background-color: rgba(0,0,0,0.03); padding: 15px; border-radius: 8px; text-align: center; border: 1px solid rgba(0,0,0,0.1);'><p style='margin-bottom: 5px; color: #555; font-size: 16px;'><b>Estatística W:</b> {stat_lc:.4f}</p><p style='margin-bottom: 0px; color: #555; font-size: 16px;'><b>Valor-p:</b> {p_formatado_lc}</p></div>", unsafe_allow_html=True)

with tab7:
    st.write("")
    st.markdown("##### Filtros Interativos")
    col_f1_t7, col_f2_t7, col_f3_t7 = st.columns(3)
    with col_f1_t7: f_raca_t7 = st.multiselect("Filtrar por Cor e Raça:", options=racas_opcoes, default=[], key="raca_t7")
    with col_f2_t7: f_regiao_t7 = st.multiselect("Filtrar por Região:", options=regioes_opcoes, default=[], key="regiao_t7")
    with col_f3_t7: f_banheiros_t7 = st.multiselect("Filtrar por Qtd Banheiros:", options=banheiros_opcoes, default=[], key="ban_t7")
    st.divider()
    st.markdown("<h6 style='color: #888;'>Distribuição Demográfica - Faixa Etária</h6>", unsafe_allow_html=True)
    
    df_idade = get_dashboard_data("tp_faixa_etaria", tuple(f_raca_t7), tuple(f_regiao_t7), tuple(f_banheiros_t7), fonte_dados)
    if not df_idade.empty:
        df_idade['categoria_str'] = df_idade['categoria'].astype(str).str.replace('.0', '', regex=False).str.strip()
        FAIXA_ETARIA_MAP = {
            '1': 'Menor de 17 anos', '2': '17 anos', '3': '18 anos', '4': '19 anos', '5': '20 anos', '6': '21 anos', '7': '22 anos', '8': '23 anos', '9': '24 anos', '10': '25 anos', '11': 'Entre 26 e 30 anos', '12': 'Entre 31 e 35 anos', '13': 'Entre 36 e 40 anos', '14': 'Entre 41 e 45 anos', '15': 'Entre 46 e 50 anos', '16': 'Entre 51 e 55 anos', '17': 'Entre 56 e 60 anos', '18': 'Entre 61 e 65 anos', '19': 'Entre 66 e 70 anos', '20': 'Maior de 70 anos'
        }
        df_idade['Faixa Etária'] = df_idade['categoria_str'].map(FAIXA_ETARIA_MAP).fillna(df_idade['categoria_str'])
        ordem_idade = list(FAIXA_ETARIA_MAP.values())

        df_idade['Faixa Etária'] = pd.Categorical(df_idade['Faixa Etária'], categories=ordem_idade, ordered=True)
        df_idade = df_idade.sort_values('Faixa Etária').dropna(subset=['Faixa Etária'])
        
        if not df_idade.empty:
            total_idade = df_idade['frequencia'].sum()
            df_idade['Freq. Absoluta'] = df_idade['frequencia']
            df_idade['Freq. Abs. Acumulada'] = df_idade['frequencia'].cumsum()
            df_idade['Freq. Relativa (%)'] = (df_idade['frequencia'] / total_idade) * 100
            df_idade['Freq. Rel. Acumulada (%)'] = df_idade['Freq. Relativa (%)'].cumsum()
            
            col_grafico, col_tabela = st.columns([1.5, 1], gap="large")
            with col_grafico:
                fig_idade = px.bar(df_idade, x='Faixa Etária', y='Freq. Relativa (%)', color_discrete_sequence=['#38B2A3'], text=df_idade['Freq. Relativa (%)'].apply(lambda x: f"{x:.1f}%"))
                fig_idade.update_traces(textposition='outside')
                fig_idade.update_layout(xaxis_title="", yaxis_title="% de Candidatos", margin=dict(t=20, b=20, l=10, r=10), height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_idade, use_container_width=True)
                
            with col_tabela:
                df_idade_display = df_idade.copy()
                df_idade_display['Freq. Absoluta'] = df_idade_display['Freq. Absoluta'].apply(lambda x: f"{int(x):,}".replace(',', '.'))
                df_idade_display['Freq. Abs. Acumulada'] = df_idade_display['Freq. Abs. Acumulada'].apply(lambda x: f"{int(x):,}".replace(',', '.'))
                df_idade_display['Freq. Relativa (%)'] = df_idade_display['Freq. Relativa (%)'].apply(lambda x: f"{x:.2f}%")
                df_idade_display['Freq. Rel. Acumulada (%)'] = df_idade_display['Freq. Rel. Acumulada (%)'].apply(lambda x: f"{x:.2f}%")
                st.dataframe(df_idade_display[['Faixa Etária', 'Freq. Absoluta', 'Freq. Abs. Acumulada', 'Freq. Relativa (%)', 'Freq. Rel. Acumulada (%)']], use_container_width=True, hide_index=True, height=450)

with tab8:
    st.write("")
    st.markdown("##### Validação Estatística da Amostra")
    st.divider()
    
    if "População" in fonte_dados:
        st.info("Você está visualizando a População Total. Para usar a validação estatística, escolha um método de amostragem na Barra Lateral (Sidebar).")
    else:
        st.write(f"Comparativo visual da distribuição de regiões entre a População Original ({st.session_state.total_participantes:,} alunos) e a sua **{fonte_dados}** ({n_amostra} alunos).".replace(',', '.'))
        
        df_pop_reg = get_dashboard_data("regiao_nome_prova", fonte="População")
        total_pop_reg = df_pop_reg['frequencia'].sum()
        df_pop_plot = df_pop_reg[['categoria', 'frequencia']].copy()
        df_pop_plot['Tipo'] = 'Brasil (População Real)'
        df_pop_plot['%'] = (df_pop_plot['frequencia'] / total_pop_reg) * 100

        df_amostra_reg = get_dashboard_data("regiao_nome_prova", fonte=fonte_dados)
        total_am_reg = df_amostra_reg['frequencia'].sum()
        df_amostra_plot = df_amostra_reg[['categoria', 'frequencia']].copy()
        df_amostra_plot['Tipo'] = 'Amostra Separada'
        df_amostra_plot['%'] = (df_amostra_plot['frequencia'] / total_am_reg) * 100

        df_comp = pd.concat([df_pop_plot, df_amostra_plot])

        fig_comp = px.bar(
            df_comp, x='categoria', y='%', color='Tipo', barmode='group',
            color_discrete_map={'Brasil (População Real)': '#5C78D8', 'Amostra Separada': '#38B2A3'},
            text=df_comp['%'].apply(lambda x: f"{x:.1f}%")
        )
        fig_comp.update_traces(textposition='outside')
        fig_comp.update_layout(
            xaxis_title="Região", yaxis_title="Representatividade (%)",
            margin=dict(t=40, b=20, l=10, r=10), height=450,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend_title_text=""
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.caption("A Amostra Estratificada garante barras perfeitamente alinhadas com a realidade. A Aleatória e Sistemática podem sofrer micro-variações normais da estatística.")
