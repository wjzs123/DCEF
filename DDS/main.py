import pandas as pd
from myfunctions_france import bitcn_model,\
    vmd_bitcn,tcn_model,dt_model,svr_model,\
    ann_model,rf_model,lstm_model,\
    emd_bitcn,eemd_bitcn,\
    ceemdan_bitcn,\
    proposed_method,\
    ceemdan_vmd_bitcn,\
    ceemdan_emd_bitcn,\
    ceemdan_eemd_bitcn



# china Domestic Aviation Ground Transport Industry Power Residential
df = pd.read_csv('dataset/china_D.csv')
# china_G,china_P,   china_D,china_R china_IA
df['Date'] = pd.to_datetime(df['date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
new_data=df[['Month','Year','Date','co2']]
new_data=new_data[new_data.Year.isin([2019, 2020, 2021,2022,2023,2024])]

cap=max(new_data['co2'])
# print(cap)
i = list(range(1, 13))
look_back=7
data_partition=0.8


dt_model(new_data, i, look_back, data_partition, cap)
svr_model(new_data,i,look_back,data_partition,cap)
ann_model(new_data,i,look_back,data_partition,cap)
rf_model(new_data,i,look_back,data_partition,cap)
lstm_model(new_data,i,look_back,data_partition,cap)
tcn_model(new_data, i, look_back, data_partition, cap)
bitcn_model(new_data, i, look_back, data_partition, cap)
vmd_bitcn(new_data, i, look_back, data_partition, cap)
emd_bitcn(new_data,i,look_back,data_partition,cap)
eemd_bitcn(new_data,i,look_back,data_partition,cap)
ceemdan_bitcn(new_data,i,look_back,data_partition,cap)
ceemdan_vmd_bitcn(new_data,i,look_back,data_partition,cap)
ceemdan_emd_bitcn(new_data,i,look_back,data_partition,cap)
ceemdan_eemd_bitcn(new_data,i,look_back,data_partition,cap)
proposed_method(new_data,i,look_back,data_partition,cap)