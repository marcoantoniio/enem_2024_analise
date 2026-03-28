import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sqlalchemy import create_engine, text
from scipy.stats import shapiro, norm

st.set_page_config(page_title="KPIs ENEM 2024", layout="wide")
st.title("📊 Indicadores Principais")
@st.cache_resource
def get_engine():
    db_config = st.secrets["database"]
    return create_engine(
        f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )

engine = get_engine()

@st.cache_data(show_spinner="Buscando opções...")
def get_filter_options():
    with engine.connect() as conn:
        racas = pd.read_sql(text("SELECT DISTINCT tp_cor_raca FROM public.ed_enem_2024_participantes WHERE tp_cor_raca IS NOT NULL AND tp_cor_raca NOT ILIKE '%issing%'"), conn)["tp_cor_raca"].tolist()
        regioes = pd.read_sql(text("SELECT DISTINCT regiao_nome_prova FROM public.ed_enem_2024_participantes WHERE regiao_nome_prova IS NOT NULL AND regiao_nome_prova NOT ILIKE '%issing%'"), conn)["regiao_nome_prova"].tolist()
        banheiros = pd.read_sql(text("SELECT DISTINCT q009 FROM public.ed_enem_2024_participantes WHERE q009 IS NOT NULL AND q009 NOT ILIKE '%issing%'"), conn)["q009"].tolist()
    return racas, regioes, banheiros

@st.cache_data(show_spinner=False)
def get_nacionalidade_data():
    query = """
        SELECT tp_nacionalidade, COUNT(*) AS frequencia
        FROM public.ed_enem_2024_participantes
        GROUP BY tp_nacionalidade;
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)

@st.cache_data(show_spinner=False)
def get_dashboard_data(column_name, racas_selecionadas=(), regioes_selecionadas=(), banheiros_selecionados=()):
    where_clauses = ["1=1"]
    where_clauses.append(f"CAST({column_name} AS TEXT) NOT ILIKE '%issing%'")
    
    if racas_selecionadas:
        raca_str = "', '".join(racas_selecionadas)
        where_clauses.append(f"tp_cor_raca IN ('{raca_str}')")
        
    if regioes_selecionadas:
        regiao_str = "', '".join(regioes_selecionadas)
        where_clauses.append(f"regiao_nome_prova IN ('{regiao_str}')")
        
    if banheiros_selecionados:
        banheiros_str = "', '".join(banheiros_selecionados)
        where_clauses.append(f"q009 IN ('{banheiros_str}')")
    
    where_sql = " AND ".join(where_clauses)
    
    query = f"""
        SELECT {column_name} AS categoria, COUNT(*) AS frequencia
        FROM public.ed_enem_2024_participantes
        WHERE {column_name} IS NOT NULL AND {where_sql}
        GROUP BY {column_name};
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
        
    if not df.empty:
        df["percentual"] = df["frequencia"] * 100.0 / df["frequencia"].sum()
    return df

@st.cache_data(show_spinner="Calculando KPIs de bens...")
def get_bens_consolidados(racas_selecionadas=(), regioes_selecionadas=()):
    where_clauses = ["1=1"]
    
    if racas_selecionadas:
        raca_str = "', '".join(racas_selecionadas)
        where_clauses.append(f"tp_cor_raca IN ('{raca_str}')")
    if regioes_selecionadas:
        regiao_str = "', '".join(regioes_selecionadas)
        where_clauses.append(f"regiao_nome_prova IN ('{regiao_str}')")
        
    where_sql = " AND ".join(where_clauses)
    
    query = f"""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN q016 = 'Sim' THEN 1 ELSE 0 END) as micro_sim,
            SUM(CASE WHEN q016 = 'Não' THEN 1 ELSE 0 END) as micro_nao,
            SUM(CASE WHEN q020 = 'Sim' THEN 1 ELSE 0 END) as wifi_sim,
            SUM(CASE WHEN q020 = 'Não' THEN 1 ELSE 0 END) as wifi_nao,
            SUM(CASE WHEN q022 = 'Não' THEN 1 ELSE 0 END) as cel_nao,
            SUM(CASE WHEN q022 = 'Sim, um' THEN 1 ELSE 0 END) as cel_1,
            SUM(CASE WHEN q022 = 'Sim, dois' THEN 1 ELSE 0 END) as cel_2,
            SUM(CASE WHEN q022 = 'Sim, três' THEN 1 ELSE 0 END) as cel_3,
            SUM(CASE WHEN q022 = 'Sim, quatro ou mais' THEN 1 ELSE 0 END) as cel_4,
            SUM(CASE WHEN q021 = 'Não' THEN 1 ELSE 0 END) as pc_nao,
            SUM(CASE WHEN q021 = 'Sim, um' THEN 1 ELSE 0 END) as pc_1,
            SUM(CASE WHEN q021 = 'Sim, dois' THEN 1 ELSE 0 END) as pc_2,
            SUM(CASE WHEN q021 = 'Sim, três' THEN 1 ELSE 0 END) as pc_3,
            SUM(CASE WHEN q021 = 'Sim, quatro ou mais' THEN 1 ELSE 0 END) as pc_4
        FROM public.ed_enem_2024_participantes
        WHERE {where_sql};
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)

@st.cache_data(show_spinner="Calculando Escolaridade dos Pais...")
def get_escolaridade_pais(racas_selecionadas=(), regioes_selecionadas=()):
    where_clauses = ["1=1"]
    if racas_selecionadas:
        raca_str = "', '".join(racas_selecionadas)
        where_clauses.append(f"tp_cor_raca IN ('{raca_str}')")
    if regioes_selecionadas:
        regiao_str = "', '".join(regioes_selecionadas)
        where_clauses.append(f"regiao_nome_prova IN ('{regiao_str}')")
    
    where_sql = " AND ".join(where_clauses)
    
    query = f"""
        SELECT 'pai' AS responsavel, q001 AS categoria, COUNT(*) AS frequencia
        FROM public.ed_enem_2024_participantes
        WHERE q001 IS NOT NULL AND CAST(q001 AS TEXT) NOT ILIKE '%issing%' AND {where_sql}
        GROUP BY q001
        UNION ALL
        SELECT 'mae' AS responsavel, q002 AS categoria, COUNT(*) AS frequencia
        FROM public.ed_enem_2024_participantes
        WHERE q002 IS NOT NULL AND CAST(q002 AS TEXT) NOT ILIKE '%issing%' AND {where_sql}
        GROUP BY q002;
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)

@st.cache_data(show_spinner="Processando distribuições (Amostra segura)...")
def get_amostra_notas():
    query = """
        SELECT 
            nota_cn_ciencias_da_natureza AS "C. da Natureza",
            nota_ch_ciencias_humanas AS "Ciências Humanas",
            nota_lc_linguagens_e_codigos AS "Linguagens",
            nota_mt_matematica AS "Matemática",
            nota_redacao AS "Redação",
            nota_media_5_notas AS "Média Geral"
        FROM public.ed_enem_2024_resultados
        LIMIT 30000;
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)

def format_kpi_value(val):
    if val == 0: return "0%"
    if val < 5: return f"{val:.1f}%"
    return f"{val:.0f}%"

def get_percentage(df, category_col, category_name):
    if df.empty: return 0.0
    try:
        return float(df.loc[df[category_col] == category_name, "percentual"].iloc[0])
    except IndexError:
        return 0.0

def render_kpi_html(valor, label, font_size="35px", label_size="14px"):
    return f"""
        <div style='text-align: center; background-color: transparent;'>
            <div style='color: #268C82; font-size: {font_size}; font-weight: bold; line-height: 1.1;'>{format_kpi_value(valor)}</div>
            <div style='color: #888; font-size: {label_size};'>{label}</div>
        </div>
    """

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Nacionalidade", 
    "Dashboard (Cor, Região e Bens)", 
    "Escolaridade dos Pais",
    "Banheiros e Renda",
    "Distribuição de Notas",
    "Normalidade (Shapiro-Wilk)",
    "Perfil Demográfico (Idade)"
])

racas_opcoes, regioes_opcoes, banheiros_opcoes = get_filter_options()

with tab1:
    st.subheader("Indicadores de Nacionalidade")
    df_nac = get_nacionalidade_data()

    if not df_nac.empty:
        total_nac = df_nac["frequencia"].sum()
        df_nac["percentual"] = df_nac["frequencia"] * 100.0 / total_nac

        estrangeiro = get_percentage(df_nac, "tp_nacionalidade", "Estrangeiro(a)")
        nao_inf = get_percentage(df_nac, "tp_nacionalidade", "Não informado")
        outros = df_nac[~df_nac["tp_nacionalidade"].isin(["Estrangeiro(a)", "Não informado"])]["percentual"].sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Estrangeiro(a)", f"{estrangeiro:.1f}%")
        col2.metric("Não informado", f"{nao_inf:.1f}%")
        col3.metric("Outros", f"{outros:.1f}%")

with tab2:
    st.write("") 
    st.markdown("##### 🔍 Filtros Interativos")
    col_filtro1, col_filtro2 = st.columns(2)
    
    with col_filtro1:
        f_raca = st.multiselect("Filtrar por Cor e Raça:", options=racas_opcoes, default=[])
    with col_filtro2:
        f_regiao = st.multiselect("Filtrar por Região:", options=regioes_opcoes, default=[])

    st.divider()
    col_esquerda, col_direita = st.columns([1, 1], gap="large")

    with col_esquerda:
        st.markdown("<h6 style='color: #888;'>Frequência por cor e raça</h6>", unsafe_allow_html=True)
        df_raca = get_dashboard_data("tp_cor_raca", (), tuple(f_regiao))
        
        if not df_raca.empty:
            fig_raca = px.pie(df_raca, values='frequencia', names='categoria', hole=0.6)
            fig_raca.update_traces(textposition='inside', textinfo='percent', rotation=90)
            fig_raca.update_layout(
                showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0),
                margin=dict(t=10, b=10, l=10, r=10), height=380,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_raca, use_container_width=True)

        st.write("") 

        st.markdown("<h6 style='color: #888;'>Frequência por região</h6>", unsafe_allow_html=True)
        df_regiao = get_dashboard_data("regiao_nome_prova", tuple(f_raca), ()) 
        
        if not df_regiao.empty:
            fig_regiao = px.treemap(df_regiao, path=['categoria'], values='frequencia', color_continuous_scale='GnBu')
            fig_regiao.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_regiao, use_container_width=True)

    with col_direita:
        df_bens = get_bens_consolidados(tuple(f_raca), tuple(f_regiao))
        
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
    st.markdown("##### 🔍 Filtros Interativos")
    col_filtro1_t3, col_filtro2_t3 = st.columns(2)
    with col_filtro1_t3:
        f_raca_t3 = st.multiselect("Filtrar por Cor e Raça:", options=racas_opcoes, default=[], key="raca_t3")
    with col_filtro2_t3:
        f_regiao_t3 = st.multiselect("Filtrar por Região:", options=regioes_opcoes, default=[], key="regiao_t3")
    st.divider()

    st.markdown("<h6 style='color: #888; text-align: center;'>Frequência por cor e raça</h6>", unsafe_allow_html=True)
    df_raca_t3 = get_dashboard_data("tp_cor_raca", (), tuple(f_regiao_t3))
    
    if not df_raca_t3.empty:
        fig_raca_t3 = px.pie(df_raca_t3, values='frequencia', names='categoria', hole=0.6)
        fig_raca_t3.update_traces(textposition='inside', textinfo='percent', rotation=90)
        fig_raca_t3.update_layout(
            showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0),
            margin=dict(t=10, b=30, l=10, r=10), height=350,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_raca_t3, use_container_width=True)

    st.write("---")
    
    df_escolaridade = get_escolaridade_pais(tuple(f_raca_t3), tuple(f_regiao_t3))
    
    col_pai, col_mae = st.columns(2, gap="large")

    with col_pai:
        st.markdown("<h6 style='color: #888;'>Nivel de Escolaridade do Pai</h6>", unsafe_allow_html=True)
        df_pai = df_escolaridade[df_escolaridade['responsavel'] == 'pai']
        
        if not df_pai.empty:
            df_pai = df_pai.sort_values(by='frequencia', ascending=True)
            df_pai['freq_formatada'] = df_pai['frequencia'].apply(lambda x: f"{x:,}".replace(',', '.'))
            fig_pai = px.bar(df_pai, x='frequencia', y='categoria', orientation='h', text='freq_formatada')
            fig_pai.update_traces(marker_color='#38B2A3', textposition='inside', insidetextanchor='end')
            fig_pai.update_layout(
                xaxis_title="", yaxis_title="", xaxis=dict(showgrid=True, showticklabels=False),
                margin=dict(l=10, r=10, t=10, b=10), height=450,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_pai, use_container_width=True)

    with col_mae:
        st.markdown("<h6 style='color: #888;'>Nivel de Escolaridade da Mãe</h6>", unsafe_allow_html=True)
        df_mae = df_escolaridade[df_escolaridade['responsavel'] == 'mae']
        
        if not df_mae.empty:
            df_mae = df_mae.sort_values(by='frequencia', ascending=True)
            df_mae['freq_formatada'] = df_mae['frequencia'].apply(lambda x: f"{x:,}".replace(',', '.'))
            fig_mae = px.bar(df_mae, x='frequencia', y='categoria', orientation='h', text='freq_formatada')
            fig_mae.update_traces(marker_color='#38B2A3', textposition='inside', insidetextanchor='end')
            fig_mae.update_layout(
                xaxis_title="", yaxis_title="", xaxis=dict(showgrid=True, showticklabels=False),
                margin=dict(l=10, r=10, t=10, b=10), height=450,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_mae, use_container_width=True)

with tab4:
    st.write("")
    st.markdown("##### 🔍 Filtros Interativos")
    f_banheiros = st.multiselect("Filtrar por Quantidade de Banheiros:", options=banheiros_opcoes, default=[], key="banheiros_t4")
    st.divider()

    st.markdown("<h6 style='color: #888; text-align: center;'>Quantos banheiros em casa</h6>", unsafe_allow_html=True)
    df_banheiros = get_dashboard_data("q009")
    if not df_banheiros.empty:
        fig_banheiros = px.pie(df_banheiros, values='frequencia', names='categoria', hole=0.6)
        fig_banheiros.update_traces(textposition='outside', textinfo='label+value+percent')
        fig_banheiros.update_layout(
            showlegend=False, margin=dict(t=80, b=80, l=50, r=50), height=450,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
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

    df_renda = get_dashboard_data("q007", banheiros_selecionados=tuple(f_banheiros))
    
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
        
        st.dataframe(
            df_renda_display[['Renda Familiar', 'Freq. Absoluta', 'Freq. Abs. Acumulada', 'Freq. Relativa (%)', 'Freq. Rel. Acumulada (%)']], 
            use_container_width=True, hide_index=True
        )

with tab5:
    st.write("")
    st.markdown("##### Comparativo Geral de Notas")
    st.write("*(As curvas mostram onde há a maior concentração de notas para cada prova)*")
    st.divider()
    
    df_amostra = get_amostra_notas()
    
    if not df_amostra.empty:
        df_melted = df_amostra.melt(var_name="Prova", value_name="Nota").dropna()
        fig_violino = px.violin(df_melted, y="Nota", x="Prova", color="Prova", box=True, points=False)
        fig_violino.update_layout(
            showlegend=False, xaxis_title="", yaxis_title="Pontuação",
            margin=dict(t=30, b=30, l=10, r=10), height=600,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_violino, use_container_width=True)

with tab6:
    st.write("")
    st.markdown("##### 📐 Teste de Normalidade Estatística")
    st.write("*(Histogramas com Curva Normal Teórica sobreposta. Amostra de até 5.000 notas)*")
    st.divider()

    df_amostra = get_amostra_notas()

    if not df_amostra.empty:
        df_teste = df_amostra[['Matemática', 'Linguagens']].dropna()
        if len(df_teste) > 5000:
            df_teste = df_teste.sample(n=5000, random_state=42)

        col_mt, col_lc = st.columns(2, gap="large")
        num_bins = 50
        raca_cor = '#38B2A3'
        lingua_cor = '#5C78D8'

        with col_mt:
            st.markdown("<h6 style='color: #888; text-align: center;'>Distribuição - Matemática</h6>", unsafe_allow_html=True)
            x = df_teste['Matemática']
            mu_mt, std_mt = x.mean(), x.std()
            fig_mt = go.Figure()
            fig_mt.add_trace(go.Histogram(x=x, nbinsx=num_bins, name='Média Geral', marker_color=raca_cor, opacity=0.75))

            x_theoretical = np.linspace(x.min(), x.max(), 200)
            y_theoretical = norm.pdf(x_theoretical, mu_mt, std_mt)
            fator_escala = len(x) * ((x.max() - x.min()) / num_bins)
            fig_mt.add_trace(go.Scatter(x=x_theoretical, y=y_theoretical * fator_escala, mode='lines', name='Curva Normal', line=dict(color='black', width=2)))

            fig_mt.update_layout(xaxis_title="Nota", yaxis_title="Frequência (Qtd de Alunos)", margin=dict(t=10, b=10, l=10, r=10), height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, bargap=0.01)
            st.plotly_chart(fig_mt, use_container_width=True)

            stat_mt, p_mt = shapiro(x)
            p_formatado_mt = f"{p_mt:.4f}" if p_mt >= 0.0001 else "< 0,0001"
            st.markdown(f"<div style='background-color: rgba(0,0,0,0.03); padding: 15px; border-radius: 8px; text-align: center; border: 1px solid rgba(0,0,0,0.1);'><p style='margin-bottom: 5px; color: #555; font-size: 16px;'><b>Estatística W:</b> {stat_mt:.4f}</p><p style='margin-bottom: 0px; color: #555; font-size: 16px;'><b>Valor-p:</b> {p_formatado_mt}</p></div>", unsafe_allow_html=True)

        with col_lc:
            st.markdown("<h6 style='color: #888; text-align: center;'>Distribuição - Linguagens</h6>", unsafe_allow_html=True)
            x = df_teste['Linguagens']
            mu_lc, std_lc = x.mean(), x.std()
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Histogram(x=x, nbinsx=num_bins, name='Linguagens', marker_color=lingua_cor, opacity=0.75))

            x_theoretical = np.linspace(x.min(), x.max(), 200)
            y_theoretical = norm.pdf(x_theoretical, mu_lc, std_lc)
            fator_escala = len(x) * ((x.max() - x.min()) / num_bins)
            fig_lc.add_trace(go.Scatter(x=x_theoretical, y=y_theoretical * fator_escala, mode='lines', name='Curva Normal', line=dict(color='black', width=2)))

            fig_lc.update_layout(xaxis_title="Nota", yaxis_title="Frequência (Qtd de Alunos)", margin=dict(t=10, b=10, l=10, r=10), height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, bargap=0.01)
            st.plotly_chart(fig_lc, use_container_width=True)

            stat_lc, p_lc = shapiro(x)
            p_formatado_lc = f"{p_lc:.4f}" if p_lc >= 0.0001 else "< 0,0001"
            st.markdown(f"<div style='background-color: rgba(0,0,0,0.03); padding: 15px; border-radius: 8px; text-align: center; border: 1px solid rgba(0,0,0,0.1);'><p style='margin-bottom: 5px; color: #555; font-size: 16px;'><b>Estatística W:</b> {stat_lc:.4f}</p><p style='margin-bottom: 0px; color: #555; font-size: 16px;'><b>Valor-p:</b> {p_formatado_lc}</p></div>", unsafe_allow_html=True)

with tab7:
    st.write("")
    st.markdown("##### 🔍 Filtros Interativos")
    col_f1_t7, col_f2_t7, col_f3_t7 = st.columns(3)
    
    with col_f1_t7:
        f_raca_t7 = st.multiselect("Filtrar por Cor e Raça:", options=racas_opcoes, default=[], key="raca_t7")
    with col_f2_t7:
        f_regiao_t7 = st.multiselect("Filtrar por Região:", options=regioes_opcoes, default=[], key="regiao_t7")
    with col_f3_t7:
        f_banheiros_t7 = st.multiselect("Filtrar por Qtd Banheiros:", options=banheiros_opcoes, default=[], key="ban_t7")
    
    st.divider()
    
    st.markdown("<h6 style='color: #888;'>Distribuição Demográfica - Faixa Etária</h6>", unsafe_allow_html=True)
    
    df_idade = get_dashboard_data("tp_faixa_etaria", tuple(f_raca_t7), tuple(f_regiao_t7), tuple(f_banheiros_t7))
    
    if not df_idade.empty:
        df_idade['categoria_str'] = df_idade['categoria'].astype(str).str.replace('.0', '', regex=False).str.strip()

        FAIXA_ETARIA_MAP = {
            '1': 'Menor de 17 anos', '2': '17 anos', '3': '18 anos', '4': '19 anos',
            '5': '20 anos', '6': '21 anos', '7': '22 anos', '8': '23 anos',
            '9': '24 anos', '10': '25 anos', '11': 'Entre 26 e 30 anos',
            '12': 'Entre 31 e 35 anos', '13': 'Entre 36 e 40 anos',
            '14': 'Entre 41 e 45 anos', '15': 'Entre 46 e 50 anos',
            '16': 'Entre 51 e 55 anos', '17': 'Entre 56 e 60 anos',
            '18': 'Entre 61 e 65 anos', '19': 'Entre 66 e 70 anos',
            '20': 'Maior de 70 anos'
        }
        
        df_idade['Faixa Etária'] = df_idade['categoria_str'].map(FAIXA_ETARIA_MAP).fillna(df_idade['categoria_str'])

        ordem_idade = [
            "Menor de 17 anos", "17 anos", "18 anos", "19 anos", "20 anos",
            "21 anos", "22 anos", "23 anos", "24 anos", "25 anos",
            "Entre 26 e 30 anos", "Entre 31 e 35 anos", "Entre 36 e 40 anos",
            "Entre 41 e 45 anos", "Entre 46 e 50 anos", "Entre 51 e 55 anos",
            "Entre 56 e 60 anos", "Entre 61 e 65 anos", "Entre 66 e 70 anos",
            "Maior de 70 anos"
        ]

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
                fig_idade = px.bar(
                    df_idade, x='Faixa Etária', y='Freq. Relativa (%)', 
                    color_discrete_sequence=['#38B2A3'],
                    text=df_idade['Freq. Relativa (%)'].apply(lambda x: f"{x:.1f}%")
                )
                fig_idade.update_traces(textposition='outside')
                fig_idade.update_layout(
                    xaxis_title="", yaxis_title="% de Candidatos",
                    margin=dict(t=20, b=20, l=10, r=10), height=450,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_idade, use_container_width=True)
                
            with col_tabela:
                df_idade_display = df_idade.copy()
                df_idade_display['Freq. Absoluta'] = df_idade_display['Freq. Absoluta'].apply(lambda x: f"{int(x):,}".replace(',', '.'))
                df_idade_display['Freq. Abs. Acumulada'] = df_idade_display['Freq. Abs. Acumulada'].apply(lambda x: f"{int(x):,}".replace(',', '.'))
                df_idade_display['Freq. Relativa (%)'] = df_idade_display['Freq. Relativa (%)'].apply(lambda x: f"{x:.2f}%")
                df_idade_display['Freq. Rel. Acumulada (%)'] = df_idade_display['Freq. Rel. Acumulada (%)'].apply(lambda x: f"{x:.2f}%")
                
                st.dataframe(
                    df_idade_display[['Faixa Etária', 'Freq. Absoluta', 'Freq. Abs. Acumulada', 'Freq. Relativa (%)', 'Freq. Rel. Acumulada (%)']], 
                    use_container_width=True, hide_index=True, height=450
                )
        else:
            st.warning("Erro na organização dos dados.")
    else:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
