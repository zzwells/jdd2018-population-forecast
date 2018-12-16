# coding: utf8
"""
STDN Model Architecture
reference:
Huaxiu Yao, Xianfeng Tang, Hua Wei, Guanjie Zheng, Zhenhui Li,
Revisiting Spatial-Temporal Similarity: A Deep Learning Framework for Traffic Prediction
"""

import tensorflow as tf
from attention import Attention

K = tf.keras.backend


def rmsle(y_true, y_pred):
    log_true = K.log(y_true + 1)
    log_pred = K.log(y_pred + 1)
    return K.sqrt(K.mean(K.square(log_pred - log_true), axis=-1))


class STDNModel:
    def __init__(self):
        pass

    def build_graph(
            self,
            long_term_seq_len,  # P for long-term periodic information
            att_cover_len,  # Q for periodically shifted attention mechanism
            short_term_seq_len,  # short-term temporal dependency
            graph_args,
            nbhd_size,
            flow_type=4,  # 4 flow features
            output_shape=45  # 预测未来15天的dwell/inflow/outflow volume
    ):

        conv_filters_num = graph_args["conv_filters_num"]
        filter_size = graph_args["filter_size"]
        cnn_flat_size = graph_args["cnn_flat_size"]
        lstm_out_size = graph_args["lstm_out_size"]
        dropout = graph_args["dropout"]
        recurrent_dropout = graph_args["recurrent_dropout"]
        activation = graph_args["activation"]
        kernel_initializer = graph_args["kernel_initializer"]
        bias_initializer = graph_args["kernel_initializer"]
        short_term_feature_vec_len = 3 * short_term_seq_len + 4   # short-term external feature size
        long_term_seq_len = long_term_seq_len - int((att_cover_len - 1) / 2)    # 实际long-term长度

        if graph_args["optimizer"] == "sgd":
            optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.005, clipnorm=5.)
        elif graph_args["optimizer"] == "adagrad":
            optimizer = tf.keras.optimizers.Adagrad(lr=0.01, decay=0.005, clipnorm=5.)
        elif graph_args["optimizer"] == "adam":
            optimizer = tf.keras.optimizers.Adam(lr=0.01, decay=0.005, clipnorm=5.)
        elif graph_args["optimizer"] == "nadam":
            optimizer = tf.keras.optimizers.Nadam(lr=0.01, schedule_decay=0.005, clipnorm=5.)
        else:
            raise NotImplementedError("invalid optimizer")

        # loss = graph_args["loss"]

        # Attention(long-term)部分（21*7个inputs）
        flat_att_flow_inputs = [tf.keras.layers.Input(
            shape=(nbhd_size, nbhd_size, flow_type,), dtype="float32",
            name="att_flow_volume_input_{0}_{1}".format(ts, att))
            for att in range(att_cover_len) for ts in range(long_term_seq_len)]

        att_flow_inputs = []
        for ts in range(long_term_seq_len):
            att_flow_inputs.append(flat_att_flow_inputs[ts * att_cover_len: (ts + 1) * att_cover_len])

        # attention(long-term) lstm external feature
        att_lstm_ex_inputs = [tf.keras.layers.Input(
            shape=(att_cover_len, short_term_feature_vec_len,), dtype="float32",
            name="att_lstm_ex_input_{0}".format(ts)) for ts in range(long_term_seq_len)]
        # short-term flow feature for cnn
        flow_inputs = [tf.keras.layers.Input(
            shape=(nbhd_size, nbhd_size, flow_type,), dtype="float32",
            name="flow_volume_input_{0}".format(ts)) for ts in range(short_term_seq_len)]
        # short-term lstm external feature
        lstm_ex_inputs = tf.keras.layers.Input(shape=(short_term_seq_len, short_term_feature_vec_len),
                                               dtype="float32", name="lstm_ex_input")

        # ====== Short-Term Part ======
        # 原文是volume cnn乘以flow cnn+activation得到的gate，这里没有volume cnn，故舍弃gate
        # # 1st convolutional layer
        # flow cnn，公式（3）
        flow_convs = [tf.keras.layers.Conv2D(
            filters=conv_filters_num, kernel_size=(filter_size, filter_size), padding="same",
            name="flow_conv_layer0_{0}".format(ts))(flow_inputs[ts]) for ts in range(short_term_seq_len)]
        flow_convs = [tf.keras.layers.Activation(
            "relu", name="flow_conv_activation_layer0_{0}".format(ts))(flow_convs[ts])
            for ts in range(short_term_seq_len)]

        # # 2nd convolutional layer
        flow_convs = [tf.keras.layers.Conv2D(
            filters=conv_filters_num, kernel_size=(filter_size, filter_size), padding="same",
            name="flow_conv_layer1_{0}".format(ts))(flow_convs[ts]) for ts in range(short_term_seq_len)]
        flow_convs = [tf.keras.layers.Activation(
            "relu", name="flow_conv_activation_layer1_{0}".format(ts))(flow_convs[ts])
            for ts in range(short_term_seq_len)]

        # After K gated convolutional layers, we use a flatten layer followed by a fully connected layer
        nbhd_vecs = [tf.keras.layers.Flatten(
            name="nbhd_flatten_layer_{0}".format(ts))(flow_convs[ts]) for ts in range(short_term_seq_len)]
        nbhd_vecs = [tf.keras.layers.Dense(
            units=cnn_flat_size,
            name="nbhd_dense_layer_{0}".format(ts))(nbhd_vecs[ts]) for ts in range(short_term_seq_len)]
        nbhd_vecs = [tf.keras.layers.Activation(
            "relu", name="nbhd_dense_activation_layer_{0}".format(ts))(nbhd_vecs[ts])
            for ts in range(short_term_seq_len)]

        # feature concatenate
        nbhd_vec = tf.keras.layers.Concatenate(axis=-1)(nbhd_vecs)
        nbhd_vec = tf.keras.layers.Reshape((short_term_seq_len, cnn_flat_size))(nbhd_vec)
        lstm_pre = tf.keras.layers.Concatenate(axis=-1)([lstm_ex_inputs, nbhd_vec])
        # lstm_ex_inputs=e[i,t], nbhd_vec=y[i,t]

        # 公式（2）
        lstm_short_term = tf.keras.layers.LSTM(
            units=lstm_out_size, return_sequences=True, activation=activation,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            dropout=dropout, recurrent_dropout=recurrent_dropout)(lstm_pre)

        # ====== Attention (Long-Term) Part ======
        # long-term feature length = valid long_term_seq_len * att_cover_len
        att_flow_convs = [[tf.keras.layers.Conv2D(
            filters=conv_filters_num, kernel_size=(filter_size, filter_size), padding="same",
            name="att_flow_conv_layer0_{0}_{1}".format(ts, att))(att_flow_inputs[ts][att])
            for att in range(att_cover_len)] for ts in range(long_term_seq_len)]
        att_flow_convs = [[tf.keras.layers.Activation(
            "relu",
            name="att_flow_conv_activation_layer0_{0}_{1}".format(ts, att))(att_flow_convs[ts][att])
            for att in range(att_cover_len)] for ts in range(long_term_seq_len)]

        att_flow_convs = [[tf.keras.layers.Conv2D(
            filters=conv_filters_num, kernel_size=(filter_size, filter_size), padding="same",
            name="att_flow_conv_layer1_{0}_{1}".format(ts, att))(att_flow_convs[ts][att])
            for att in range(att_cover_len)] for ts in range(long_term_seq_len)]
        att_flow_convs = [[tf.keras.layers.Activation(
            "relu",
            name="att_flow_conv_activation_layer1_{0}_{1}".format(ts, att))(att_flow_convs[ts][att])
            for att in range(att_cover_len)] for ts in range(long_term_seq_len)]

        att_nbhd_vecs = [[tf.keras.layers.Flatten(
            name="att_nbhd_flatten_layer_{0}_{1}".format(ts, att))(att_flow_convs[ts][att])
            for att in range(att_cover_len)] for ts in range(long_term_seq_len)]
        att_nbhd_vecs = [[tf.keras.layers.Dense(
            units=cnn_flat_size,
            name="att_nbhd_dense_layer_{0}_{1}".format(ts, att))(att_nbhd_vecs[ts][att])
            for att in range(att_cover_len)] for ts in range(long_term_seq_len)]
        att_nbhd_vecs = [[tf.keras.layers.Activation(
            "relu",
            name="att_nbhd_dense_activation_layer_{0}_{1}".format(ts, att))(att_nbhd_vecs[ts][att])
            for att in range(att_cover_len)] for ts in range(long_term_seq_len)]

        att_nbhd_vec = [tf.keras.layers.Concatenate(axis=-1)(att_nbhd_vecs[ts])
                        for ts in range(long_term_seq_len)]
        att_nbhd_vec = [tf.keras.layers.Reshape((att_cover_len, cnn_flat_size))(att_nbhd_vec[ts])
                        for ts in range(long_term_seq_len)]
        # att_lstm_ex_inputs[att]=e[p,q][i,t], att_nbhd_vec=y[p,q][i,t]
        att_lstm_pre = [tf.keras.layers.Concatenate(axis=-1)([att_lstm_ex_inputs[ts], att_nbhd_vec[ts]])
                        for ts in range(long_term_seq_len)]
        # 公式（5）
        att_lstms = [tf.keras.layers.LSTM(
            units=lstm_out_size, return_sequences=True, activation=activation,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            dropout=dropout, recurrent_dropout=recurrent_dropout,
            name="att_lstm_{0}".format(att))(att_lstm_pre[att]) for att in range(long_term_seq_len)]

        # Implement the Attention
        # 调用call(inputs=[att_lstms[att], lstm])，每一天(P)算一下所有Q的attention，算到公式（7）
        att_level_1 = [Attention(method='cba')([att_lstms[ts], lstm_short_term]) for ts in range(long_term_seq_len)]
        att_level_1 = tf.keras.layers.Concatenate(axis=-1)(att_level_1)
        att_level_1 = tf.keras.layers.Reshape((long_term_seq_len, lstm_out_size))(att_level_1)
        # 公式（9）
        att_level_2 = tf.keras.layers.LSTM(
            units=lstm_out_size, return_sequences=True, activation=activation,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            dropout=dropout, recurrent_dropout=recurrent_dropout)(att_level_1)
        # Joint Training
        # shape [(None, 21, 128), (None, 7, 128)]，一个30天的特征，一个7天的特征，合并成(None, 28, 128)
        merge_all = tf.keras.layers.Concatenate(axis=1)([att_level_2, lstm_short_term])
        # merge_all = Dropout(rate=0.5)(merge_all)
        # Dense层输入输出的维数要一致，输出是（None, 45），输入要reshape
        merge_all = tf.keras.layers.Reshape((merge_all.get_shape()[-1]*merge_all.get_shape()[-2], ))(merge_all)
        pred_out = tf.keras.layers.Dense(units=output_shape, activation="softmax")(merge_all)   # 公式（10）

        # Model(inputs=)不接受[[Input(...)]]只接受[Input(...)]，嵌套列表全部压平
        feed_inputs = flat_att_flow_inputs + att_lstm_ex_inputs + flow_inputs + [lstm_ex_inputs]
        model = tf.keras.models.Model(inputs=feed_inputs, outputs=pred_out)
        model.compile(optimizer=optimizer, loss=rmsle, metrics=[rmsle])
        return model
