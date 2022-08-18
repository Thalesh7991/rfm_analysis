import pandas as pd
import streamlit as st
import numpy as np
import folium
import plotly.graph_objects as go
import plotly.express as px

# Setando o tamanho da página
st.set_page_config( layout='wide' )

# Leitura dos Dados
df_raw = pd.read_csv('data/Ecommerce.csv', encoding='unicode_escape')
df_raw = df_raw.drop('Unnamed: 8',axis=1)

def pct_rank_qcut(series, n, tipo_analise):
    if tipo_analise == 'Recency':
        edges = pd.Series([float(i) / n for i in range(n + 1)])
        f = lambda x: (edges >= x).values.argmax()
        return series.rank(pct=1, ascending=0).apply(f)
    else:
        edges = pd.Series([float(i) / n for i in range(n + 1)])
        f = lambda x: (edges >= x).values.argmax()
        return series.rank(pct=1).apply(f)


# Define rfm_level function
def rfm_level(df):
    if df['R'] <= 2 and df['F'] <= 2.5:
        return 'Hibernando'
    elif (df['R'] <= 2 and df['F'] > 2.5 and df[
        'F'] <= 4.5):  # Clientes que ja tiveram uma frequencia alta, porém não compram faz muito tempo
        return 'Em Risco'
    elif (df['R'] <= 2 and df[
        'F'] > 4.5):  # Clientes que ja tiveram uma frequencia alta, porém não compram faz muito tempo
        return 'Prestes a Perder'
    elif (df['R'] > 2 and df['R'] <= 3.5 and df['F'] <= 2.5):
        return 'Prestes a Dormir'

    elif (df['R'] > 2.5 and df['R'] <= 3.5 and df['F'] > 2.5 and df['F'] <= 3.5):
        return 'Precisa de Atenção'

    elif (df['R'] > 2 and df['R'] < 5 and df['F'] > 3.5):
        return 'Clientes Fiéis'

    elif (df['R'] > 3.5 and df['R'] <= 4.5 and df['F'] <= 1.5):
        return 'Promissores'

    elif (df['R'] > 3.5 and df['F'] >= 1.5 and df['F'] <= 3.5):
        return 'Potentiais Clientes Fieis'
    elif (df['R'] >= 5 and df['F'] >= 3.5):
        return 'Champions'
    elif (df['R'] >= 5 and df['F'] <= 1.5):
        return 'Novos Clientes'


# Dropando Clientes com código NA
df_raw = df_raw.dropna(subset=['CustomerID'])

#Change Data Type
df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate'])


#Data Filtering
# Quantidade Negativa
df2 = df_raw.loc[df_raw['Quantity'] > 0]


# stock_code
df2 = df2[~df2['StockCode'].isin( ['POST', 'D', 'DOT', 'M', 'S', 'AMAZONFEE', 'm', 'DCGSSBOY', 'DCGSSGIRL', 'PADS', 'B', 'CRUK'] ) ]

#Bad User
df2 =  df2.loc[~df2['CustomerID'].isin([16446])]

##### FEATURE ENGINEERING #####
df3 = df2.copy()


df_ref = df3.drop(['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice','Country'],axis=1).drop_duplicates( ignore_index=True )

#faturamento
df3['faturamento'] = df3['Quantity'] * df3['UnitPrice']
df_monetary = df3[['CustomerID', 'faturamento']].groupby('CustomerID').sum().reset_index()
df_ref = pd.merge(df_ref, df_monetary, how='left', on='CustomerID')


#recencia
df_recency = df3[['CustomerID', 'InvoiceDate']].groupby('CustomerID').max().reset_index()
df_recency['recency_days'] = (df3['InvoiceDate'].max() - df_recency['InvoiceDate']).dt.days
df_recency = df_recency[['CustomerID','recency_days']].copy()
df_ref = pd.merge(df_ref, df_recency, how='left', on='CustomerID')

#frequencia
df_aux = ( df3[['CustomerID', 'InvoiceNo', 'InvoiceDate']].drop_duplicates()
                                                             .groupby( 'CustomerID')
                                                             .agg( max_ = ( 'InvoiceDate', 'max' ),
                                                                   min_ = ( 'InvoiceDate', 'min' ),
                                                                   days_= ( 'InvoiceDate', lambda x: ( ( x.max() - x.min() ).days ) + 1 ),
                                                                   buy_ = ( 'InvoiceNo', 'count' ) ) ).reset_index()
# Frequency
df_aux['frequency'] = df_aux[['buy_', 'days_']].apply( lambda x: x['buy_'] / x['days_'] if  x['days_'] != 0 else 0, axis=1 )

# Merge
df_ref = pd.merge( df_ref, df_aux[['CustomerID', 'frequency']], on='CustomerID', how='left' )


# rankeando cada atributo
df_ref['R'] = pct_rank_qcut(df_ref['recency_days'],5, 'Recency')
df_ref['F'] = pct_rank_qcut(df_ref['frequency'],5, 'Frequency')
df_ref['M'] = pct_rank_qcut(df_ref['faturamento'],5, 'Monetary')

# Aplicando SEGMENTAÇÃO DOS CLIENTES
df_ref['segmentation'] = df_ref.apply(rfm_level, axis=1 )






#print(df_ref.dtypes)

#c1,c2 = st.columns( (1,1) )
#c1.header('Header Coluna 1')

#with c1:
#    st.dataframe(df_teste)

with st.sidebar:
    st.title('FILTROS')


    fat_min = df_ref['faturamento'].min()
    fat_max = df_ref['faturamento'].max()
    fat_mean = float(df_ref['faturamento'].mean())

    rec_min = df_ref['recency_days'].min()
    rec_max = df_ref['recency_days'].max()
    rec_mean = float(df_ref['recency_days'].mean())

    freq_min = df_ref['frequency'].min()
    freq_max = df_ref['frequency'].max()
    freq_mean = float(df_ref['frequency'].mean())




    #Filtro de Grupo
    groups_list = df_ref['segmentation'].drop_duplicates()
    group_filter = st.multiselect('Grupo', groups_list)
    if group_filter:
        group_filter = group_filter
    else:
        group_filter = groups_list

    #Filtro de Cliente
    customers_list = df_ref['CustomerID'].drop_duplicates()
    customer_filter = st.multiselect('Cliente', customers_list)
    if customer_filter:
        customer_filter = customer_filter
    else:
        customer_filter = customers_list

    # Filtro de R Score
    r_list = df_ref['R'].drop_duplicates()
    r_score_filter = st.multiselect('Recency Score', r_list)
    if r_score_filter:
        r_score_filter = r_score_filter
    else:
        r_score_filter = r_list

    # Filtro de Frequency Score
    f_list = df_ref['F'].drop_duplicates()
    f_score_filter = st.multiselect('Frequency Score', f_list)
    if f_score_filter:
        f_score_filter = f_score_filter
    else:
        f_score_filter = f_list

    # Filtro de Monetary Score
    m_list = df_ref['M'].drop_duplicates()
    m_score_filter = st.multiselect('Score', m_list)
    if m_score_filter:
        m_score_filter = m_score_filter
    else:
        m_score_filter = m_list

    #Filtro de faturamento
    fat_slider = st.slider('Faturamento', min_value= float(fat_min), max_value= float(fat_max), step=1.0, value=float(fat_max) )



    # Filtro de Recência
    rec_slider = st.slider('Recência', min_value= float(rec_min), max_value= float(rec_max), step=1.0, value=float(rec_max) )

    # Filtro de Frequência
    freq_slider = st.slider('Frequência', min_value= float(freq_min), max_value= float(freq_max), step=1.0, value=float(freq_max) )


st.title("Dashboard de Perfil dos Clientes")
print(df_ref.columns)

df_ref = df_ref.loc[(df_ref['CustomerID'].isin(customer_filter)
                     & df_ref['segmentation'].isin(group_filter)
                     & df_ref['R'].isin(r_score_filter)
                     & df_ref['F'].isin(f_score_filter)
                     & df_ref['M'].isin(m_score_filter)
                     & (df_ref['faturamento'] <= fat_slider)
                     & (df_ref['recency_days'] <= rec_slider)
                     & (df_ref['frequency'] <= freq_slider)

                     )]

# Pefil dos Grupos
df_segmentation = df_ref[['CustomerID','segmentation']].groupby('segmentation').count().reset_index()
df_segmentation['perc_customer'] = (df_segmentation['CustomerID'] / df_segmentation['CustomerID'].sum() )*100
df_fat = df_ref[['segmentation','faturamento']].groupby('segmentation').mean().reset_index()
df_segmentation =  pd.merge(df_segmentation, df_fat, on='segmentation', how='left')
df_rec = df_ref[['segmentation','recency_days']].groupby('segmentation').mean().reset_index()
df_segmentation =  pd.merge(df_segmentation, df_rec, on='segmentation', how='left')
df_freq = df_ref[['segmentation','frequency']].groupby('segmentation').mean().reset_index()
df_segmentation =  pd.merge(df_segmentation, df_freq, on='segmentation', how='left')
df_segmentation.sort_values('faturamento', ascending=False)

new_columns = ['Grupo', 'Qtde Clientes', '% de Clientes', 'Faturamento', 'Recência', 'Frequência']
df_segmentation.columns = new_columns

st.dataframe(df_segmentation.loc[ (df_segmentation['Grupo'].isin(group_filter)
                                   #& df_segmentation['']

                                   ) ])

#st.dataframe(df_segmentation.sort_values('faturamento', ascending=0 ))
#####



#####



c1, c2 = st.columns(2)
c1.header('Clientes por Grupo')

#with c1:
# Bar chart Customer per Segmentation
customer_per_group = df_ref[['CustomerID', 'segmentation']].groupby(
    'segmentation').count().reset_index().sort_values('CustomerID', ascending=0)
fig = px.bar(customer_per_group, x='segmentation', y='CustomerID',
                labels={
                    "segmentation": "Grupos",
                    "CustomerID": "Qtde Clientes"
                }
             )
c1.plotly_chart(fig,width=50)

#with c2:
# Bar chart Faturamento per Segmentation
c2.header('Faturamento Médio')
faturamento_per_group = df_ref[['segmentation', 'faturamento']].groupby(
    'segmentation').mean().reset_index().sort_values('faturamento', ascending=0)
fig2 = px.bar(faturamento_per_group, x='segmentation', y='faturamento',
                labels={
                    "segmentation": "Grupos",
                    "faturamento": "Faturamento Médio (R$)"
                }
              )
c2.plotly_chart(fig2,use_container_width=True)


c1, c2 = st.columns(2)
# Bar chart recency per Segmentation
c1.header('Tempo Médio Sem Compra (Dias)')
recencia_per_group = df_ref[['segmentation', 'recency_days']].groupby(
    'segmentation').mean().reset_index().sort_values('recency_days', ascending=0)
fig3 = px.bar(recencia_per_group, x='segmentation', y='recency_days',
                labels={
                    "segmentation": "Grupos",
                    "recency_days": "Tempo Sem Comprar"
                }

              )
c1.plotly_chart(fig3, use_container_width=True)

c2.header('Frequência de Compra')
frequencia_per_group = df_ref.loc[(df_ref['faturamento'] <= fat_slider)
                                   & (df_ref['recency_days'] <= rec_slider)
                                   & (df_ref['frequency'] <= freq_slider)
                                   & (df_ref['CustomerID'].isin(customer_filter))
                                   & (df_ref['segmentation'].isin(group_filter))][['segmentation', 'frequency']].groupby(
    'segmentation').mean().reset_index().sort_values('frequency', ascending=0)
fig4 = px.bar(frequencia_per_group, x='segmentation', y='frequency',
                labels={
                    "segmentation": "Grupos",
                    "frequency": "Frequência"
                }
              )
c2.plotly_chart(fig4,use_container_width=True)




# Linha Temporal Faturamento Médio por Mês
#print(df_raw.columns)
df3['TotalFat'] = df3['UnitPrice'] * df3['Quantity']
df3 = pd.merge(df3, df_ref, on='CustomerID', how='left')
#df3 = df3.loc[ (df3['segmentation'].isin(group_filter))]
df_fat = df3.loc[(  df3['segmentation'].isin(group_filter) )

                ][['TotalFat', 'InvoiceDate']]

st.header("Faturamento Mensal")

fig5 = px.line(df_fat.groupby(pd.Grouper(key='InvoiceDate', axis=0, freq='M')).sum().reset_index(), x='InvoiceDate', y='TotalFat',
                labels={
                    "InvoiceDate": "Meses",
                    "TotalFat": "Faturamento"
                }
               )
st.plotly_chart(fig5,use_container_width=True,
                )
#st.dataframe(df_fat.groupby(pd.Grouper(key='InvoiceDate', axis=0, freq='M')).sum().reset_index())


df_filtered = df_ref = df_ref.loc[ (df_ref['faturamento'] <= fat_slider)
                                   & (df_ref['recency_days'] <= rec_slider)
                                   & (df_ref['frequency'] <= freq_slider)
                                   & (df_ref['CustomerID'].isin(customer_filter))
                                   & (df_ref['segmentation'].isin(group_filter))
                                   & (df_ref['R'].isin(r_score_filter))
                                   & (df_ref['F'].isin(f_score_filter))
                                   & (df_ref['M'].isin(m_score_filter))
                                 ]

st.header("Relatório de Clientes")
df_filtered.columns = ['Cod Cliente', 'Faturamento', 'Recência (dias)', 'Frequência', 'R', 'F', 'M', 'Grupo']
st.dataframe(df_filtered.sort_values('Faturamento', ascending=0 ))



