import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import random
from scipy.stats import shapiro, norm

st.set_page_config(page_title="Dashboard ENEM - Amostragem", layout="wide")

# Nomes dos arquivos Parquet
FILE_P = "'dados_enem_participantes_pt*.parquet'"
FILE_R = "'dados_enem_resultados_pt*.parquet'"

# =========================
# CONFIGURAÇÃO DA BARRA LATERAL E AMOSTRAGEM
# =========================
st.sidebar.image("logo_enem.png", width=150)
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
st.sidebar.write(f"**Nível de Confiança:** 95%\n\n**Margem de Erro:** 1%\n\n**Tamanho Alvo:** {n_amostra:,}".replace(',', '.'))

# Pega o total da população apenas uma vez
@st.cache_data
def get_totais():
    t_p = duckdb.query(f"SELECT COUNT(*) FROM {FILE_P}").fetchone()[0]
    t_r = duckdb.query(f"SELECT COUNT(*) FROM {FILE_R}").fetchone()[0]
    return t_p, t_r

total_populacao_p, total_populacao_r = get_totais()

# =========================
# GERADOR DE AMOSTRAS (DuckDB SQL)
# =========================
@st.cache_data(show_spinner=False)
def gerar_amostra(tipo_amostra, tamanho):
    if "População" in tipo_amostra:
        return None, None
    
    with st.spinner(f"Gerando {tipo_amostra}..."):
        if tipo_amostra == "Amostra Aleatória Simples":
            df_p = duckdb.query(f"SELECT * FROM {FILE_P} USING SAMPLE reservoir({tamanho} ROWS)").df()
            df_r = duckdb.query(f"SELECT * FROM {FILE_R} USING SAMPLE reservoir({tamanho} ROWS)").df()
            return df_p, df_r
            
        elif tipo_amostra == "Amostra Sistemática":
            k_p = total_populacao_p // tamanho
            start_p = random.randint(1, k_p)
            q_p = f"SELECT * FROM (SELECT *, row_number() OVER () as rn FROM {FILE_P}) WHERE rn % {k_p} = {start_p} LIMIT {tamanho}"
            df_p = duckdb.query(q_p).df()
            
            k_r = total_populacao_r // tamanho
            start_r = random.randint(1, k_r)
            q_r = f"SELECT * FROM (SELECT *, row_number() OVER () as rn FROM {FILE_R}) WHERE rn % {k_r} = {start_r} LIMIT {tamanho}"
            df_r = duckdb.query(q_r).df()
            return df_p, df_r
            
        elif tipo_amostra == "Amostra Estratificada":
            proporcoes = duckdb.query(f"SELECT regiao_nome_prova, COUNT(*) as qtd FROM {FILE_P} WHERE regiao_nome_prova IS NOT NULL GROUP BY regiao_nome_prova").df()
            total_validos = proporcoes['qtd'].sum()
            
            dfs_p = []
            for _, row in proporcoes.iterrows():
                regiao = row['regiao_nome_prova']
                n_estrato = int(round((row['qtd'] / total_validos) * tamanho))
                if n_estrato > 0:
                    q = f"SELECT * FROM (SELECT * FROM {FILE_P} WHERE regiao_nome_prova = '{regiao}') USING SAMPLE reservoir({n_estrato} ROWS)"
                    dfs_p.append(duckdb.query(q).df())
                    
            df_p = pd.concat(dfs_p, ignore_index=True)
            
            # Garantir que a base de resultados puxe exatamente a mesma quantidade gerada na estratificada
            tamanho_real = len(df_p)
            df_r = duckdb.query(f"SELECT * FROM {FILE_R} USING SAMPLE reservoir({tamanho_real} ROWS)").df()
            return df_p, df_r

df_ativo_p, df_ativo_r = gerar_amostra(fonte_dados, n_amostra)

SOURCE_P = FILE_P if "População" in fonte_dados else "df_ativo_p"
SOURCE_R = FILE_R if "População" in fonte_dados else "df_ativo_r"

st.title(f"📊 Indicadores Principais - {fonte_dados}")


# =========================
# FUNÇÕES DE FILTRAGEM BLINDADAS
# =========================
def formatar_para_sql(lista_valores):
    """Função blindada: Pega a lista do Streamlit e converte perfeitamente para SQL -> 'A', 'B' """
    if not lista_valores: return ""
    # Escapa aspas simples caso existam no nome e formata com aspas por fora
    valores_seguros = [str(v).replace("'", "''") for v in lista_valores]
    return ", ".join([f"'{v}'" for v in valores_seguros])


@st.cache_data(show_spinner=False)
def get_filter_options():
    racas = duckdb.query(f"SELECT DISTINCT tp_cor_raca FROM {FILE_P} WHERE tp_cor_raca IS NOT NULL AND CAST(tp_cor_raca AS VARCHAR) NOT ILIKE '%issing%'").df()['tp_cor_raca'].tolist()
    regioes = duckdb.query(f"SELECT DISTINCT regiao_nome_prova FROM {FILE_P} WHERE regiao_nome_prova IS NOT NULL AND CAST(regiao_nome_prova AS VARCHAR) NOT ILIKE '%issing%'").df()['regiao_nome_prova'].tolist()
    banheiros = duckdb.query(f"SELECT DISTINCT q009 FROM {FILE_P} WHERE q009 IS NOT NULL AND CAST(q009 AS VARCHAR) NOT ILIKE '%issing%'").df()['q009'].tolist()
    return sorted(racas), sorted(regioes), sorted(banheiros)

def get_dashboard_data(column_name, racas_selecionadas=None, regioes_selecionadas=None, banheiros_selecionados=None):
    racas_selecionadas = racas_selecionadas or []
    regioes_selecionadas = regioes_selecionadas or []
    banheiros_selecionados = banheiros_selecionados or []
    
    where_clauses = ["1=1", f"{column_name} IS NOT NULL", f"CAST({column_name} AS VARCHAR) NOT ILIKE '%issing%'"]
    
    if racas_selecionadas: 
        where_clauses.append(f"tp_cor_raca IN ({formatar_para_sql(racas_selecionadas)})")
    if regioes_selecionadas: 
        where_clauses.append(f"regiao_nome_prova IN ({formatar_para_sql(regioes_selecionadas)})")
    if banheiros_selecionados: 
        where_clauses.append(f"q009 IN ({formatar_para_sql(banheiros_selecionados)})")
        
    where_str = " AND ".join(where_clauses)
    query = f"SELECT {column_name} AS categoria, COUNT(*) AS frequencia FROM {SOURCE_P} WHERE {where_str} GROUP BY {column_name};"
    
    df = duckdb.query(query).df()
    if not df.empty: 
        df["percentual"] = df["frequencia"] * 100.0 / df["frequencia"].sum()
    return df

def get_bens_consolidados(racas_selecionadas=None, regioes_selecionadas=None):
    racas_selecionadas = racas_selecionadas or []
    regioes_selecionadas = regioes_selecionadas or []
    
    where_clauses = ["1=1"]
    if racas_selecionadas: 
        where_clauses.append(f"tp_cor_raca IN ({formatar_para_sql(racas_selecionadas)})")
    if regioes_selecionadas: 
        where_clauses.append(f"regiao_nome_prova IN ({formatar_para_sql(regioes_selecionadas)})")
        
    where_str = " AND ".join(where_clauses)
    
    query = f"""
        SELECT COUNT(*) as total,
        SUM(CASE WHEN q016 = 'Sim' THEN 1 ELSE 0 END) as micro_sim, SUM(CASE WHEN q016 = 'Não' THEN 1 ELSE 0 END) as micro_nao,
        SUM(CASE WHEN q020 = 'Sim' THEN 1 ELSE 0 END) as wifi_sim, SUM(CASE WHEN q020 = 'Não' THEN 1 ELSE 0 END) as wifi_nao,
        SUM(CASE WHEN q022 = 'Não' THEN 1 ELSE 0 END) as cel_nao, SUM(CASE WHEN q022 = 'Sim, um' THEN 1 ELSE 0 END) as cel_1,
        SUM(CASE WHEN q022 = 'Sim, dois' THEN 1 ELSE 0 END) as cel_2, SUM(CASE WHEN q022 = 'Sim, três' THEN 1 ELSE 0 END) as cel_3,
        SUM(CASE WHEN q022 = 'Sim, quatro ou mais' THEN 1 ELSE 0 END) as cel_4, SUM(CASE WHEN q021 = 'Não' THEN 1 ELSE 0 END) as pc_nao,
        SUM(CASE WHEN q021 = 'Sim, um' THEN 1 ELSE 0 END) as pc_1, SUM(CASE WHEN q021 = 'Sim, dois' THEN 1 ELSE 0 END) as pc_2,
        SUM(CASE WHEN q021 = 'Sim, três' THEN 1 ELSE 0 END) as pc_3, SUM(CASE WHEN q021 = 'Sim, quatro ou mais' THEN 1 ELSE 0 END) as pc_4
        FROM {SOURCE_P} WHERE {where_str};
    """
    return duckdb.query(query).df()

def get_escolaridade_pais(racas_selecionadas=None, regioes_selecionadas=None):
    racas_selecionadas = racas_selecionadas or []
    regioes_selecionadas = regioes_selecionadas or []
    
    where_clauses = ["1=1"]
    if racas_selecionadas: 
        where_clauses.append(f"tp_cor_raca IN ({formatar_para_sql(racas_selecionadas)})")
    if regioes_selecionadas: 
        where_clauses.append(f"regiao_nome_prova IN ({formatar_para_sql(regioes_selecionadas)})")
        
    where_str = " AND ".join(where_clauses)
    
    query = f"""
        SELECT 'pai' AS responsavel, q001 AS categoria, COUNT(*) AS frequencia FROM {SOURCE_P} WHERE q001 IS NOT NULL AND CAST(q001 AS VARCHAR) NOT ILIKE '%issing%' AND {where_str} GROUP BY q001
        UNION ALL
        SELECT 'mae' AS responsavel, q002 AS categoria, COUNT(*) AS frequencia FROM {SOURCE_P} WHERE q002 IS NOT NULL AND CAST(q002 AS VARCHAR) NOT ILIKE '%issing%' AND {where_str} GROUP BY q002;
    """
    return duckdb.query(query).df()

def get_amostra_notas():
    # Sampling blindado para o Plotly não travar
    query = f"""
        SELECT nota_cn_ciencias_da_natureza AS "C. da Natureza", 
               nota_ch_ciencias_humanas AS "Ciências Humanas", 
               nota_lc_linguagens_e_codigos AS "Linguagens", 
               nota_mt_matematica AS "Matemática", 
               nota_redacao AS "Redação", 
               nota_media_5_notas AS "Média Geral" 
        FROM {SOURCE_R} 
        USING SAMPLE reservoir(30000 ROWS);
    """
    return duckdb.query(query).df()


# =========================
# UI HELPER FUNCTIONS
# =========================
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


# =========================
# INTERFACE DO USUÁRIO (TABS)
# =========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Nacionalidade", "Dashboard (Cor, Região, Bens)", "Escolaridade Pais",
    "Banheiros e Renda", "Distribuição de Notas", "Normalidade (Shapiro-Wilk)",
    "Perfil Demográfico", "Validação da Amostra"
])

racas_opcoes, regioes_opcoes, banheiros_opcoes = get_filter_options()

with tab1:
    st.subheader("Indicadores de Nacionalidade")
    df_nac = get_dashboard_data("tp_nacionalidade")
    
    if not df_nac.empty:
        estrangeiro = get_percentage(df_nac, "categoria", "Estrangeiro(a)")
        nao_inf = get_percentage(df_nac, "categoria", "Não informado")
        outros = df_nac[~df_nac["categoria"].isin(["Estrangeiro(a)", "Não informado"])]["percentual"].sum()

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
        df_raca = get_dashboard_data("tp_cor_raca", regioes_selecionadas=f_regiao)
        if not df_raca.empty:
            fig_raca = px.pie(df_raca, values='frequencia', names='categoria', hole=0.6)
            fig_raca.update_traces(textposition='inside', textinfo='percent', rotation=90)
            fig_raca.update_layout(showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0), margin=dict(t=10, b=10, l=10, r=10), height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_raca, use_container_width=True)

        st.markdown("<h6 style='color: #888;'>Frequência por região</h6>", unsafe_allow_html=True)
        df_regiao_dash = get_dashboard_data("regiao_nome_prova", racas_selecionadas=f_raca) 
        if not df_regiao_dash.empty:
            fig_regiao = px.treemap(df_regiao_dash, path=['categoria'], values='frequencia', color_continuous_scale='GnBu')
            fig_regiao.update_layout(margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_regiao, use_container_width=True)

    with col_direita:
        df_bens = get_bens_consolidados(racas_selecionadas=f_raca, regioes_selecionadas=f_regiao)
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
    df_raca_t3 = get_dashboard_data("tp_cor_raca", regioes_selecionadas=f_regiao_t3)
    if not df_raca_t3.empty:
        fig_raca_t3 = px.pie(df_raca_t3, values='frequencia', names='categoria', hole=0.6)
        fig_raca_t3.update_traces(textposition='inside', textinfo='percent', rotation=90)
        fig_raca_t3.update_layout(showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0), margin=dict(t=10, b=30, l=10, r=10), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_raca_t3, use_container_width=True)

    st.write("---")
    df_escolaridade = get_escolaridade_pais(racas_selecionadas=f_raca_t3, regioes_selecionadas=f_regiao_t3)
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
    df_banheiros_dash = get_dashboard_data("q009")
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

    df_renda = get_dashboard_data("q007", banheiros_selecionados=f_banheiros)
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
    
    df_amostra_notas = get_amostra_notas()
    
    if not df_amostra_notas.empty:
        df_melted = df_amostra_notas.melt(var_name="Prova", value_name="Nota").dropna()
        fig_violino = px.violin(df_melted, y="Nota", x="Prova", color="Prova", box=True, points=False)
        fig_violino.update_layout(showlegend=False, xaxis_title="", yaxis_title="Pontuação", margin=dict(t=30, b=30, l=10, r=10), height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_violino, use_container_width=True)

with tab6:
    st.write("")
    st.markdown("##### Teste de Normalidade Estatística")
    st.divider()
    
    df_teste = duckdb.query(f'SELECT nota_mt_matematica AS "Matemática", nota_lc_linguagens_e_codigos AS "Linguagens" FROM {SOURCE_R} WHERE nota_mt_matematica IS NOT NULL AND nota_lc_linguagens_e_codigos IS NOT NULL USING SAMPLE reservoir(5000 ROWS)').df()
    
    if not df_teste.empty:
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
    
    df_idade = get_dashboard_data("tp_faixa_etaria", racas_selecionadas=f_raca_t7, regioes_selecionadas=f_regiao_t7, banheiros_selecionados=f_banheiros_t7)
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
        qtd_amostra = len(df_ativo_p)
        st.write(f"Comparativo visual da distribuição de regiões entre a População Original ({total_populacao_p:,} alunos) e a sua **{fonte_dados}** ({qtd_amostra} alunos).".replace(',', '.'))
        
        q_pop = f"SELECT regiao_nome_prova AS categoria, COUNT(*) AS frequencia FROM {FILE_P} WHERE regiao_nome_prova IS NOT NULL AND CAST(regiao_nome_prova AS VARCHAR) NOT ILIKE '%issing%' GROUP BY regiao_nome_prova;"
        df_pop_plot = duckdb.query(q_pop).df()
        df_pop_plot['Tipo'] = 'Brasil (População Real)'
        df_pop_plot['%'] = (df_pop_plot['frequencia'] / df_pop_plot['frequencia'].sum()) * 100

        q_amostra = "SELECT regiao_nome_prova AS categoria, COUNT(*) AS frequencia FROM df_ativo_p WHERE regiao_nome_prova IS NOT NULL AND CAST(regiao_nome_prova AS VARCHAR) NOT ILIKE '%issing%' GROUP BY regiao_nome_prova;"
        df_amostra_plot = duckdb.query(q_amostra).df()
        df_amostra_plot['Tipo'] = 'Amostra Separada'
        df_amostra_plot['%'] = (df_amostra_plot['frequencia'] / df_amostra_plot['frequencia'].sum()) * 100

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
