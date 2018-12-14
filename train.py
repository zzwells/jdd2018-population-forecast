# coding: utf8

import os
import logging
import tensorflow as tf
import numpy as np
# import pickle
import json
from features import FeatureFactory
from model import STDNModel
K = tf.keras.backend

try:
    import psutil
    psutil_installed = True
except ImportError:
    psutil_installed = False


class CustomStopper(tf.keras.callbacks.EarlyStopping):
    def __init__(self, monitor='val_rmsle', min_delta=0, patience=5, mode='min', start_epoch=5):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


class STDNTrainer:
    def __init__(self):
        self.config = json.load(open("config.json", "r", encoding="utf-8"))
        self.dates = np.load(self.config["date_list"])
        self.att_cover_len = self.config["attention_cover_daynum"]
        self.long_term_seq_len = self.config["long_term_sequence_len"]
        self.short_term_seq_len = self.config["short_term_sequence_len"]
        self.cnn_nbhd_size = self.config["cnn_neighbor_size"]
        self.history_feature_daynum = self.config["history_feature_daynum"]
        self.forecast_ahead_daynum = self.config["forecast_ahead_daynum"]

        self.batch_size = self.config["batch_size"]
        self.epochs = self.config["epochs"]
        self.patience = self.config["patience_earlystop"]
        # early_stop = CustomStopper(patience=patience)

        self.sampler = FeatureFactory(self.config["volume_train"], self.config["flow_train"])
        self.modeler = STDNModel()

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    def rmsle(self, y_true, y_pred):
        log_true = K.log(y_true + 1)
        log_pred = K.log(y_pred + 1)
        return K.sqrt(K.mean(K.square(log_pred - log_true)))

    def inverse_rmsle(self, x, n_sample):
        return np.power(x, 2) * n_sample

    def memory_monitor(self, ps):
        mem_used = ps.memory_full_info().uss / 1024 / 1024 / 1024
        logging.warning("Memory used: {:.2f} GB".format(mem_used))

    def memory_monitor_v2(self):
        """ Linux服务器可能无法安装psutil """
        with open('/proc/meminfo') as f:
            mem_total = int(f.readline().split()[1])
            mem_free = int(f.readline().split()[1])
            _ = f.readline()   # 跳过一行
            mem_buffer = int(f.readline().split()[1])
            mem_cache = int(f.readline().split()[1])
        mem_used = mem_total - mem_free - mem_buffer - mem_cache
        mem_used = mem_used / 1024 / 1024 / 1024
        logging.warning("Memory used: {:.2f} GB".format(mem_used))

    def gen_input_to_model(self, att_flow_inputs, att_lstm_ex_inputs, flow_inputs, lstm_ex_inputs):
        final_input = {}
        valid_long_term_seq_len = self.long_term_seq_len - int((self.att_cover_len - 1) / 2)
        for ts in range(valid_long_term_seq_len):
            final_input["att_lstm_ex_input_{0}".format(ts)] = att_lstm_ex_inputs[:, ts]
            for att in range(self.att_cover_len):
                final_input["att_flow_volume_input_{0}_{1}".format(ts, att)] = att_flow_inputs[:, ts, att]

        for ts in range(self.short_term_seq_len):
            final_input["flow_volume_input_{0}".format(ts)] = flow_inputs[:, ts]

        final_input["lstm_ex_input"] = lstm_ex_inputs

        return final_input

    def train(self):

        model = self.modeler.build_graph(
            graph_args=self.config, long_term_seq_len=self.long_term_seq_len, att_cover_len=self.att_cover_len,
            short_term_seq_len=self.short_term_seq_len, nbhd_size=2*self.cnn_nbhd_size+1,
            output_shape=3*self.forecast_ahead_daynum)

        init_start = self.long_term_seq_len + 1   # 最前面的序列要用来取特征
        losses = []
        best_loss = 0
        no_improve = 0
        stop_flag = False
        batch_start_idx = int(init_start)
        for ep in range(self.epochs):
            logging.info("********** Epoch %d Begins **********" % (ep+1))
            epoch_end = False
            loss_epoch = []
            step = 0
            while not epoch_end:
                batch_end_idx = int(batch_start_idx + self.batch_size + self.forecast_ahead_daynum)
                if batch_end_idx >= len(self.dates) - 1:
                    n_dates_batch = len(self.dates[batch_start_idx:])
                    epoch_end = True
                else:
                    n_dates_batch = batch_end_idx - batch_start_idx

                logging.info(
                    "Sampling from date {0} to {1} ...".format(
                        self.dates[batch_start_idx], self.dates[batch_start_idx + n_dates_batch])
                )

                x_att_flow, x_att_lstm_ex, x_flow, x_short_term, y = self.sampler.sample_stdn(
                    mode="train", dates=self.dates, date_cursor=batch_start_idx,
                    n_dates_batch=n_dates_batch, label_daynum=self.forecast_ahead_daynum,
                    long_term_seq_len=self.long_term_seq_len, att_cover_len=self.att_cover_len,
                    short_term_seq_len=self.short_term_seq_len, history_feature_daynum=self.history_feature_daynum,
                    cnn_nbhd_size=self.cnn_nbhd_size)

                logging.info("Sample Completed")
                # logging.info("Attention Flow CNN Feature {0}".format(x_att_flow.shape))    # (6272, 21, 7, 7, 7, 4)
                # logging.info("Attention LSTM External Feature {0}".format(x_att_lstm_ex.shape))  # (6272, 21, 7, 25)
                # logging.info("Short-Term Flow CNN Feature {0}".format(x_flow.shape))    # (6272, 7, 7, 7, 4)
                # logging.info("Short-Term External Feature {0}".format(x_short_term.shape))    # (6272, 7, 25)
                # logging.info("Labels {0}".format(y.shape))     # (6272, 45)
                # logging.info("+" * 50)

                # cache_data = list()
                # cache_data.append(x_att_flow)
                # cache_data.append(x_att_lstm_ex)
                # cache_data.append(x_flow)
                # cache_data.append([x_short_term, ])
                # cache_data.append(y)
                # pickle.dump(cache_data, open(config["saved_input_features"], "wb"))
                # np.save(config["saved_input_features"], np.array(cache_data))

                # model.fit(x=[x_att_flow, x_att_lstm_ex, x_flow, x_short_term], y=y,
                #           batch_size=batch_size, validation_split=config["validation_fraction"],
                #           epochs=epochs, shuffle=False, callbacks=[early_stop])

                feed_x = self.gen_input_to_model(x_att_flow, x_att_lstm_ex, x_flow, x_short_term)
                loss_step = model.train_on_batch(feed_x, y)
                logging.info("training loss at iter {} of epoch {} : {}".format(step, ep, loss_step))
                loss_epoch.append(self.inverse_rmsle(loss_step, self.forecast_ahead_daynum))
                if batch_end_idx >= len(self.dates):
                    losses.append(np.sqrt(np.mean(loss_epoch)))   # loss of whole epoch
                    if len(losses) == self.patience:
                        best_loss = min(losses)
                    elif len(losses) > self.patience:
                        best_loss_tmp = min(losses)
                        if best_loss_tmp >= best_loss:
                            no_improve += 1
                        else:
                            best_loss = best_loss_tmp
                            no_improve = 0
                            model.save(
                                self.config["saved_model"] + "_epoch" + str(ep+1) + "_loss" + str(best_loss) + ".hdf5")
                    if no_improve >= self.patience:
                        stop_flag = True

                if psutil_installed:
                    ps = psutil.Process(os.getpid())
                    self.memory_monitor(ps)
                else:
                    self.memory_monitor_v2()

                batch_start_idx += self.batch_size
                if batch_start_idx >= len(self.dates) - 1:
                    epoch_end = True
                step += 1

            if stop_flag:
                break

        best_epoch = losses.index(min(losses))
        model = tf.keras.models.load_model(
            self.config["saved_model"] + "_epoch" + str(best_epoch+1) + "_loss" + str(best_loss) + ".hdf5")

        # 给出最后一条样本的预测，去掉这条样本本身就是未来15天
        x_att_flow, x_att_lstm_ex, x_flow, x_short_term, y = self.sampler.sample_stdn(
            mode='prediction', dates=self.dates, date_cursor=len(self.dates)-1,
            n_dates_batch=1, label_daynum=self.forecast_ahead_daynum,
            long_term_seq_len=self.long_term_seq_len, att_cover_len=self.att_cover_len,
            short_term_seq_len=self.short_term_seq_len, history_feature_daynum=self.history_feature_daynum,
            cnn_nbhd_size=self.cnn_nbhd_size)
        feed_x = self.gen_input_to_model(x_att_flow, x_att_lstm_ex, x_flow, x_short_term)
        preds = model.predict(feed_x)
        logging.info("prediction result: {0}".format(preds))
        np.save(self.config["saved_prediction"], preds)
        # pickle.dump(preds, open(config["saved_prediction"], "wb"))
        # preds = pickle.load(open(config["saved_prediction"], "wb"))


if __name__ == "__main__":
    STDNTrainer().train()
