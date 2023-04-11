#%%

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from netdata_pandas.data_cloud import get_data_cloud, get_charts_cloud
import plotly.express as px
import re

#%%

st.title('Netdata Clustered Heatmap!')

#%%

# defaults
netdata_api_token = st.secrets.netdata_api_token
default_space_id = 'ea93d7b8-0df6-45c0-b13d-1560996c89eb'
default_room_id = 'd8a4e0c5-7c79-4145-900e-83a9f06fcb6a'

# inputs
netdata_api_token = st.sidebar.text_input('netdata_api_token', value=netdata_api_token)
space_id = st.sidebar.text_input('space_id', value=default_space_id)
room_id = st.sidebar.text_input('room_id', value=default_room_id)
node_id_list = st.sidebar.text_input('node_id_list', value='').split(',')
contexts_regex = st.sidebar.text_input('contexts_regex', value='system\..*')
after = st.sidebar.number_input('after', value=-60*10)
before = st.sidebar.number_input('before', value=0)
freq = st.sidebar.text_input('freq', value='10s')
n_clusters = st.sidebar.number_input('n_clusters', value=5)
fig_w = st.sidebar.number_input('fig_w', value=900)
fig_h = st.sidebar.number_input('fig_h', value=25)

#%%

charts_cloud = get_charts_cloud(space_id, room_id)
contexts_cloud = list(set([charts_cloud[c]['context'] for c in charts_cloud]))

pattern = re.compile(contexts_regex)
contexts_matched = [context for context in contexts_cloud if pattern.match(context)]

#%%

# get data from netdata cloud
df = pd.DataFrame(columns=['time'])
for context in contexts_matched:
    try:
        df_context = get_data_cloud(space_id, room_id, context, after, before, freq=freq)
        df_context = df_context.add_prefix(f'{context}.')
    except:
        df_context = pd.DataFrame(columns=['time'])
    
    df = df.merge(df_context,how='outer',on='time')

df = df.set_index('time')

print(df.shape)

#%%

# normalize each column to be 0 to 1
# https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
df = ( df-df.min() ) / ( df.max() - df.min() )

# ffill and bfill any missing data
df = df.ffill().bfill()

# get X matrix to feed into clustering
X = df.transpose().dropna().values

#%%

# cluster the data
kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(X)

# sort based on clustering
cols_sorted = pd.DataFrame(
            zip(df.columns, kmeans.labels_),
            columns=['metric', 'cluster']
        ).sort_values('cluster')['metric'].values.tolist()

df = df[cols_sorted]

fig = px.imshow(df.transpose(), color_continuous_scale='Greens')
fig.update_layout(
            autosize=False,
            width=fig_w,
            height=len(df.columns)*fig_h)
st.plotly_chart(fig)

#%%

#%%