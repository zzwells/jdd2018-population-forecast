# coding: utf8
"""
生成STDN的输入feature
"""

import numpy as np
from datetime import datetime


class FeatureFactory:
    def __init__(self, volume_data_path, flow_data_path):
        self.volume_data = np.load(volume_data_path)
        self.flow_data = np.load(flow_data_path)

    def sample_stdn(self, mode, dates, date_cursor, n_dates_batch, label_daynum,
                    long_term_seq_len=30,  # P=30 for long-term periodic information
                    att_cover_len=7,   # Q=7 for periodically shifted attention mechanism (-3,-2,-1,0,1,2,3)
                    short_term_seq_len=7,  # short-term temporal dependency
                    history_feature_daynum=6,   # last 7 days as feature
                    cnn_nbhd_size=3):    # cnn feature neighborhood

        # 每次多出15个日期是为了取未来15天作为y label
        valid_num_t = n_dates_batch - label_daynum
        # 往回最后3天不能取，否则+1，+2，+3天会取到当前t的未来数据
        valid_att_num_t = long_term_seq_len - int((att_cover_len - 1) / 2)   # 24-3=21

        att_lstm_ex_features = []
        att_flow_features = []
        short_term_ex_features = []
        short_term_flow_features = []
        labels = []

        ex_feature_samples = []
        flow_feature_samples = []

        # t: 遍历样本的每个时刻; back_t: 对t回溯short_term_seq_len个时刻过程中的某个时刻
        for t in range(valid_num_t):  # 最后15天不取feature
            for x in range(self.flow_data.shape[1]):    # 98*98, 当前的目标区县为x, 即坐标(x, x)
                ex_feature_samples.clear()
                flow_feature_samples.clear()
                for seqn in range(short_term_seq_len):
                    # 实际预测时不知道当前时刻t的volume和flow，所以最多取到t-1时刻的特征
                    back_t = t - (short_term_seq_len - seqn)
                    # ====== features for flow data ======
                    cursor = date_cursor + back_t
                    flow_feature_curr_out = self.flow_data[cursor, x, :]    # x->y outflow
                    flow_feature_curr_in = self.flow_data[cursor, :, x]     # x->y in flow
                    flow_feature_last_out_to_curr = self.flow_data[cursor-1, x, :]    # 前一个时刻的outflow
                    flow_feature_curr_in_from_last = self.flow_data[cursor-1, :, x]   # 前一个时刻的inflow

                    flow_feature = np.zeros(flow_feature_curr_in.shape + (4,), dtype=np.float32)   # shape=(98, 4)
                    flow_feature[:, 0] = flow_feature_curr_out
                    flow_feature[:, 1] = flow_feature_curr_in
                    flow_feature[:, 2] = flow_feature_last_out_to_curr
                    flow_feature[:, 3] = flow_feature_curr_in_from_last

                    # 每个region考虑周围7*7个neighbor
                    local_flow_feature = np.zeros((2 * cnn_nbhd_size + 1, 2 * cnn_nbhd_size + 1, 4), dtype=np.float32)
                    # 取出这个neighborhood区域
                    for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                        # boundary check
                        if not (0 <= cnn_nbhd_x < self.flow_data.shape[1]):
                            continue
                        # 取历史每个时刻的7*7个neighbors的4个flow_feature作为feature
                        local_flow_feature[cnn_nbhd_x-(x-cnn_nbhd_size), :] = flow_feature[cnn_nbhd_x, :]

                    flow_feature_samples.append(local_flow_feature)

                    # ====== short term features for lstm ======
                    # 下面这些features相当与论文中的external features，e[i,t]
                    feature_vec = np.array([])

                    # 当前t的short-term back_t往前history_feature_daynum天(包括back_t)范围内的volume直接做feature
                    history_feature = self.volume_data[(cursor - history_feature_daynum): (cursor + 1), x, :]
                    history_feature = history_feature.flatten()
                    feature_vec = np.concatenate((feature_vec, history_feature))

                    # 日期特征
                    datetime_feature = list()
                    # 星期几
                    datetime_feature.append(datetime.strptime(str(dates[date_cursor+t]), "%Y%m%d").weekday())
                    datetime_feature.append(datetime.strptime(str(dates[cursor]), "%Y%m%d").weekday())
                    # 几号
                    datetime_feature.append(int(str(dates[date_cursor+t])[-2:]))
                    datetime_feature.append(int(str(dates[cursor])[-2:]))

                    feature_vec = np.concatenate((feature_vec, np.array(datetime_feature)))
                    ex_feature_samples.append(feature_vec)

                short_term_flow_features.append(flow_feature_samples)
                short_term_ex_features.append(ex_feature_samples)

                # ====== Periodically Shifted Attention (long term features) ======
                att_lstm_ex_features.append([])    # t*x维度+1
                att_flow_features.append([])
                ex_feature_samples.clear()
                flow_feature_samples.clear()
                for att_lstm_i in range(valid_att_num_t):
                    # 保证不能取到t后面的未来数据; 可以取到t，因为label是从t+1开始，而特征不能取到t+1
                    ex_feature_samples.clear()
                    flow_feature_samples.clear()
                    # 回溯到att_lstm_i天前，取该天的(-3,-2,-1,0,1,2,3)天，att_t指向该天+3
                    att_t = t - (valid_att_num_t - att_lstm_i) + (att_cover_len - 1) / 2   # 最小-18
                    att_t = int(att_t)
                    for seqn in range(att_cover_len):
                        back_t = att_t - (att_cover_len - seqn) + 1   # 可以取到att_t本身 (-24,-23,-22,-21,-20,-19,-18)
                        # 一样的flow feature，只是取的时刻变了
                        cursor = date_cursor + back_t
                        flow_feature_curr_out = self.flow_data[cursor, x, :]
                        flow_feature_curr_in = self.flow_data[cursor, :, x]
                        flow_feature_last_out_to_curr = self.flow_data[cursor-1, x, :]
                        flow_feature_curr_in_from_last = self.flow_data[cursor-1, :, x]

                        flow_feature = np.zeros(flow_feature_curr_in.shape + (4,), dtype=np.float32)
                        flow_feature[:, 0] = flow_feature_curr_out
                        flow_feature[:, 1] = flow_feature_curr_in
                        flow_feature[:, 2] = flow_feature_last_out_to_curr
                        flow_feature[:, 3] = flow_feature_curr_in_from_last

                        # local cnn for flow
                        local_flow_feature = np.zeros((2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, 4), dtype=np.float32)
                        for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                            # boundary check
                            if not (0 <= cnn_nbhd_x < self.flow_data.shape[1]):
                                continue
                            # 取7*7个neighbors的4个flow_feature
                            local_flow_feature[cnn_nbhd_x - (x - cnn_nbhd_size), :] = flow_feature[cnn_nbhd_x, :]

                        flow_feature_samples.append(local_flow_feature)

                        # ======= short term features for lstm (Short-term Temporal Dependency) ======
                        # 下面这些features相当与论文中的external features，e[i,t]
                        feature_vec = np.array([])

                        # 当前t的attention back_t往前history_feature_daynum天(包括back_t)范围内的volume直接做feature
                        history_feature = self.volume_data[(cursor - history_feature_daynum):(cursor + 1), x, :]
                        history_feature = history_feature.flatten()
                        feature_vec = np.concatenate((feature_vec, history_feature))

                        # 日期特征
                        datetime_feature = list()
                        datetime_feature.append(datetime.strptime(str(dates[date_cursor+t]), "%Y%m%d").weekday())
                        datetime_feature.append(datetime.strptime(str(dates[cursor]), "%Y%m%d").weekday())
                        datetime_feature.append(int(str(dates[date_cursor+t])[-2:]))
                        datetime_feature.append(int(str(dates[cursor])[-2:]))

                        feature_vec = np.concatenate((feature_vec, np.array(datetime_feature)))
                        ex_feature_samples.append(feature_vec)

                    # (t*x, valid_att_num_t, att_cover_len, nbhd_size, nbhd_size, 4)
                    att_flow_features[t * self.flow_data.shape[1] + x].append(flow_feature_samples)
                    # (t*x, valid_att_num_t, att_cover_len, 25)
                    att_lstm_ex_features[t * self.flow_data.shape[1] + x].append(ex_feature_samples)

                # label(3个回归值，该时刻t+1~t+15实际的volume)
                # shape=(98*64, 15*3)=(6272, 45)，对应LSTM输入shape=(6272, long\short-term time steps, feature_dim)
                if mode == "train":
                    labels.append(self.volume_data[(t+1):(t+label_daynum+1), x, :].flatten())

        att_flow_features = np.array(att_flow_features)
        att_lstm_ex_features = np.array(att_lstm_ex_features)
        short_term_flow_features = np.array(short_term_flow_features)
        short_term_ex_features = np.array(short_term_ex_features)
        labels = np.array(labels)

        return (att_flow_features,      # attention(long-term) flow cnn feature
                att_lstm_ex_features,        # attention(long-term) volume external feature
                short_term_flow_features,    # short-term flow cnn feature
                short_term_ex_features,      # short-term volume external_feature
                labels)                      # 回归值，每个时刻实际的volume
