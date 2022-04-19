import numpy as np
import pandas as pd
import xarray as xr

from pgportfolio.marketdata.replaybuffer import ReplayBuffer
from pgportfolio.tools.configprocess import parse_time
from pgportfolio.tools.data import get_type_list, panel_fillna


class DataMatricesNew:
    def __init__(self, dataset_file, batch_size=50, buffer_bias_ratio=0,
                 window_size=50, feature_number=3, test_portion=0.15,
                 portion_reversed=False, is_permed=False):
        """
        :param window_size: periods of input data
        :param is_permed: if False, the sample inside a mini-batch is in order
        :param test_portion: portion of test set
        :param portion_reversed: if False, the order to sets are [train, validation, test]
        else the order is [test, validation, train]
        """

        da = xr.open_dataarray(dataset_file)
        da = da.drop_sel({'features': 'volume'})
        self.__global_data = panel_fillna(da, "both")

        # assert window_size >= MIN_NUM_PERIOD
        self.__coin_no = len(self.__global_data.coins)
        type_list = get_type_list(feature_number)
        self.__features = type_list
        self.feature_number = feature_number

        # portfolio vector memory, [time, assets]
        self.__PVM = pd.DataFrame(index=self.__global_data.time,
                                  columns=self.__global_data.coins)
        self.__PVM = self.__PVM.fillna(1.0 / self.__coin_no)

        self._window_size = window_size
        self._num_periods = len(self.__global_data.time)
        self.__divide_data(test_portion, portion_reversed)

        self._portion_reversed = portion_reversed
        self.__is_permed = is_permed

        self.__batch_size = batch_size
        self.__delta = 0  # the count of global increased
        end_index = self._train_ind[-1]
        self.__replay_buffer = ReplayBuffer(start_index=self._train_ind[0],
                                            end_index=end_index,
                                            sample_bias=buffer_bias_ratio,
                                            batch_size=self.__batch_size,
                                            coin_number=self.__coin_no,
                                            is_permed=self.__is_permed)

        print("the number of training examples is %s"
              ", of test examples is %s" % (self._num_train_samples, self._num_test_samples))
        print("the training set is from %s to %s" % (min(self._train_ind), max(self._train_ind)))
        print("the test set is from %s to %s" % (min(self._test_ind), max(self._test_ind)))

    @property
    def global_weights(self):
        return self.__PVM

    @staticmethod
    def create_from_config(config):
        """main method to create the DataMatrices in this project
        @:param config: config dictionary
        @:return: a DataMatrices object
        """
        config = config.copy()
        input_config = config["input"]
        train_config = config["training"]
        start = parse_time(input_config["start_date"])
        end = parse_time(input_config["end_date"])
        return DataMatrices(start=start,
                            end=end,
                            market=input_config["market"],
                            feature_number=input_config["feature_number"],
                            window_size=input_config["window_size"],
                            online=input_config["online"],
                            period=input_config["global_period"],
                            coin_filter=input_config["coin_number"],
                            is_permed=input_config["is_permed"],
                            buffer_bias_ratio=train_config["buffer_biased"],
                            batch_size=train_config["batch_size"],
                            volume_average_days=input_config["volume_average_days"],
                            test_portion=input_config["test_portion"],
                            portion_reversed=input_config["portion_reversed"],
                            )

    @property
    def global_matrix(self):
        return self.__global_data

    @property
    def coin_list(self):
        return self.__history_manager.coins

    @property
    def num_train_samples(self):
        return self._num_train_samples

    @property
    def test_indices(self):
        return self._test_ind[:-(self._window_size + 1):]

    @property
    def num_test_samples(self):
        return self._num_test_samples

    def append_experience(self, online_w=None):
        """
        :param online_w: (number of assets + 1, ) numpy array
        Let it be None if in the backtest case.
        """
        self.__delta += 1
        self._train_ind.append(self._train_ind[-1] + 1)
        appended_index = self._train_ind[-1]
        self.__replay_buffer.append_experience(appended_index)

    def get_test_set(self):
        return self.__pack_samples(self.test_indices)

    def get_test_set_online(self, ind_start, ind_end, x_window_size):
        return self.__pack_samples_test_online(ind_start, ind_end, x_window_size)

    def get_training_set(self):
        return self.__pack_samples(self._train_ind[:-self._window_size])

    ##############################################################################
    def next_batch(self):
        """
        @:return: the next batch of training sample. The sample is a dictionary
        with key "X"(input data); "y"(future relative price); "last_w" a numpy array
        with shape [batch_size, assets]; "w" a list of numpy arrays list length is
        batch_size
        """
        batch = self.__pack_samples([exp.state_index for exp in self.__replay_buffer.next_experience_batch()])
        #        print(np.shape([exp.state_index for exp in self.__replay_buffer.next_experience_batch()]),[exp.state_index for exp in self.__replay_buffer.next_experience_batch()])
        return batch

    def __pack_samples(self, indexs):
        indexs = np.array(indexs)
        last_w = self.__PVM.values[indexs - 1, :]

        def setw(w):
            self.__PVM.iloc[indexs, :] = w

        #            print("set w index from %d-%d!" %( indexs[0],indexs[-1]))
        M = [self.get_submatrix(index) for index in indexs]
        M = np.array(M)
        X = M[:, :, :, :-1]
        y = M[:, :, :, -1] / M[:, 0, None, :, -2]
        return {"X": X, "y": y, "last_w": last_w, "setw": setw}

    def __pack_samples_test_online(self, ind_start, ind_end, x_window_size):
        #        indexs = np.array(indexs)
        last_w = self.__PVM.values[ind_start - 1:ind_start, :]

        #        y_window_size = window_size-x_window_size
        def setw(w):
            self.__PVM.iloc[ind_start, :] = w

        #            print("set w index from %d-%d!" %( indexs[0],indexs[-1]))
        M = [self.get_submatrix_test_online(ind_start, ind_end)]  # [1,4,11,2807]
        M = np.array(M)
        X = M[:, :, :, :-1]
        y = M[:, :, :, x_window_size:] / M[:, 0, None, :, x_window_size - 1:-1]
        return {"X": X, "y": y, "last_w": last_w, "setw": setw}

    ##############################################################################################
    def get_submatrix(self, ind):
        return self.__global_data.values[:, :, ind:ind + self._window_size + 1]

    def get_submatrix_test_online(self, ind_start, ind_end):
        return self.__global_data.values[:, :, ind_start:ind_end]

    def __divide_data(self, test_portion, portion_reversed):
        train_portion = 1 - test_portion
        s = float(train_portion + test_portion)
        if portion_reversed:
            portions = np.array([test_portion]) / s
            portion_split = (portions * self._num_periods).astype(int)
            indices = np.arange(self._num_periods)
            self._test_ind, self._train_ind = np.split(indices, portion_split)
        else:
            portions = np.array([train_portion]) / s
            portion_split = (portions * self._num_periods).astype(int)
            indices = np.arange(self._num_periods)
            self._train_ind, self._test_ind = np.split(indices, portion_split)

        self._train_ind = self._train_ind[:-(self._window_size + 1)]
        # NOTE(zhengyao): change the logic here in order to fit both
        # reversed and normal version
        self._train_ind = list(self._train_ind)
        self._num_train_samples = len(self._train_ind)
        self._num_test_samples = len(self.test_indices)
