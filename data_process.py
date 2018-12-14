# coding: utf8
"""
JDD数据预处理
"""

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
# 在notebook使用tqdm_notebook，防止进度条变成多行

# dwell\flow_in\flow_out数据
dat_vol = pd.read_csv('F:\\JDD2018\\flow_train.csv', header=0)
# 按date_dt, city_code, district_code排序
dat_vol.sort_values(by=['date_dt', 'city_code', 'district_code'], ascending=True, inplace=True)
# 城市数和每个城市下的区县数
citys = list(set(dat_vol['city_code']))   # 7个城市
n_district_every_city = dat_vol[['city_code', 'district_code']].groupby(['city_code'], as_index=False)\
                        .agg(lambda x: len(x.value_counts()))
n_district_every_city = dict(zip(n_district_every_city.iloc[:, 0], n_district_every_city.iloc[:, 1]))
# json.dump(n_district_every_city, open('F:\\JDD2018\\city_districts_cnt.json', 'w'))  # 11 is not serializable
max_n_district = max(n_district_every_city.values())    # 22
dates_all = dat_vol['date_dt'].unique()
np.save('F:\\JDD2018\\dates', dates_all)
# 按城市+区县排列的区县列表
districts = dat_vol.loc[dat_vol['date_dt'] == 20170601, 'district_code'].tolist()
n_district_total = len(districts)    # 98个区县


# 每一天作为一个时刻t，构造98*98矩阵，包含[dwell,flow_in,flow_out]三个维度，在矩阵对角线上
def transform_vol(rawdat, dates):
    """ output shape = (n_dates, n_district_total, 3) """
    n_dates = len(dates)
    output = np.zeros((n_dates, n_district_total, 3))
    for i_dt in range(n_dates):
        for i_dst in range(n_district_total):
            output[i_dt, i_dst, :]\
                = rawdat.loc[(rawdat['date_dt'] == dates[i_dt]) & (rawdat['district_code'] == districts[i_dst]),
                             ['dwell', 'flow_in', 'flow_out']].values
    return np.array(output)


# 训练集和验证集，验证集用于early stopping
# train_vol = transform_vol(dat_vol.loc[dat_vol['date_dt'] < 20180101, :], dates_all[dates_all < 20180101], 0)
# valid_vol = transform_vol(dat_vol.loc[dat_vol['date_dt'] >= 20180101, :], dates_all[dates_all >= 20180101], 0)
train_vol = transform_vol(dat_vol, dates_all)
print(train_vol.shape)    # (274, 98, 3)

np.save('F:\\JDD2018\\volume_transformed_train', train_vol)


""" 把所有区县排在一起当作地图
    但实际地理排布是不知道的，还有个想法是把所有城市区县做个全排列
    每种排列方式当作一个地图输入网络，最终每种排布的输出结果取平均或者再经过一个全连接层
"""

# flow data
dat_flow = pd.read_csv('F:\\JDD2018\\transition_train.csv', header=0)
dat_flow.sort_values(by=['date_dt', 'o_city_code', 'o_district_code', 'd_city_code', 'd_district_code'],
                     ascending=True, inplace=True)


def transform_flow(rawdat, dates, default_value):
    """ output shape = (n_dates, n_districts, n_districts), 构成flow matrix """
    n_dates = len(dates)
    output = []
    for i_dt in tqdm(range(n_dates)):
        tmparray = rawdat.loc[rawdat['date_dt'] == dates[i_dt], :]
        tmparray = tmparray.pivot(index='o_district_code', columns='d_district_code', values='cnt')
        tmparray = tmparray.reindex_axis(districts, axis=0)
        tmparray = tmparray.reindex_axis(districts, axis=1)   # 调整行列，统一按districts中的顺序
        tmparray.fillna(default_value, inplace=True)
        output.append(tmparray.values)
    return np.array(output)


train_flow = transform_flow(dat_flow, dates_all, 0)
print(train_flow.shape)    # (274, 98, 98)

np.save('F:\\JDD2018\\flow_transformed_train', train_flow)
