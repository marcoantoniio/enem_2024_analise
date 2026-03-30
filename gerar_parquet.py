import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
import time

# Acessa as credenciais do seu secrets.toml
db_config = st.secrets["database"]
engine = create_engine(f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")

query_participantes = """
SELECT 
    nu_inscricao, tp_nacionalidade, tp_cor_raca, regiao_nome_prova, 
    q001, q002, q007, q009, q016, q020, q021, q022, tp_faixa_etaria
FROM public.ed_enem_2024_participantes
"""

query_resultados = """
SELECT 
    nu_sequencial, nota_cn_ciencias_da_natureza, nota_ch_ciencias_humanas, 
    nota_lc_linguagens_e_codigos, nota_mt_matematica, nota_redacao, nota_media_5_notas
FROM public.ed_enem_2024_resultados
"""

print("1/2 - Baixando tabela de Participantes... (Isso pode demorar um pouco)")
df_p = pd.read_sql(query_participantes, engine)
print("Salvando participantes em Parquet...")
df_p.to_parquet("dados_enem_participantes.parquet", engine='pyarrow', index=False)

print("\n2/2 - Baixando tabela de Resultados...")
df_r = pd.read_sql(query_resultados, engine)
print("Salvando resultados em Parquet...")
df_r.to_parquet("dados_enem_resultados.parquet", engine='pyarrow', index=False)

print("\nPronto! Os dois arquivos Parquet foram gerados com sucesso.")