import numpy as np


class Trip(object):
    """
    The Trip object contains the time-series related to a single trip

    Args:
        signal_list (list of numpy.array): List of  1-dimensional arrays where each array corresponds to a time-series
        of a signal or measurement of that trip. The size of all arrays must match, otherwise an exception will be
        raised.
        trip_id (str): ID for the trip
        timestamps (list of float): list wit the timestamp of each observation
        labels_list (list of list): List of labels related to the each time-series observation. Each list inside
        labels_list relates to the one particular label for the trip (e.g., aggressiveness or type of maneuver). Thus,
        each list inside label_list needs to have the same size as each signal. This input defaults to None, in which
        case it is assume that the trips has no labels.

    Attributes:
        n_dim (int): dimension of the trip (i.e. number fo signals)
        id (str): ID for the trip
        signals (list of numpy.array): list with the trip data. Each item of the list is a 1-dimensional array
        labels (list of list): List of labels related to the each time-series observation
        timestamps (list of float): list with the timestamp of each observation
        representing a single signal or measurement of that trip
        n_obs (int): length of the signals list, i.e. the number of measurements of the trip.

    """

    def __init__(self, signal_list, trip_id, timestamps=None, labels_list=None):
        """
        Constructor for the class Trips.

        Args:
            signal_list (list of numpy.array): List of  1-dimensional arrays where each array corresponds to a time-series
        of a signal or measurement of that trip
            trip_id (int): ID of the trip
            label_list (list): List of labels related to the each time-series observation
        """
        n_obs_list = [len(signal) for signal in signal_list]
        if len(set(n_obs_list)) != 1:
            raise Exception(
                "The size of the signals is inconsistent. "
                "Make sure all signals have the same number of observations"
            )
        self.n_obs = n_obs_list[0]
        self.id = trip_id
        self.signals = signal_list
        self.n_dim = len(signal_list)
        self.labels = labels_list
        if labels_list is not None:
            for inside_list in labels_list:
                if len(inside_list) != self.n_obs:
                    raise Exception(
                        "The size of the labels_list is inconsistent with the trip size. "
                        "Make sure all list of labels provided have the same number of observations as the signals"
                    )
        if timestamps is None:
            self.timestamps = list(np.arange(self.n_obs))
        else:
            if len(timestamps) != self.n_obs:
                raise Exception(
                    "The size of the timestamps is inconsistent with the trip size. "
                    "Make sure the list of timestamps provided has the same number of observations as the signals"
                )
            else:
                self.timestamps = timestamps

    def get_single_obs(self, pointer):
        """
        Args:
            pointer (int): pointer to the trip, indicating the place from which we want to take the observation.

        Returns:
            obs_array (numpy.array): array with the multidimensional observation in the original trip. It take a single
            observation from the trip's time-series in all its dimensions.
        """
        obs_list = []
        for signal in self.signals:
            signal_obs = signal[pointer]
            obs_list.append(signal_obs)
        obs_array = np.array(obs_list)
        return obs_array

    def get_windown_obs(self, pointers):
        """
        Args:
            pointers (list of int): pointers to the trip, indicating the place from which we want to take the
            observations.

        Returns:
            windown_obs (list numpy.array): list of arrays with the multidimensional observations in the original trip.
            Each entry of the list corresponds to a single observation taken from the trip's time-series and thus, the
            length of the list is equal to the length of the pointers' list.
        """
        windown_obs = []
        for pointer in pointers:
            point_obs = self.get_single_obs(pointer)
            windown_obs.append(point_obs)
        return windown_obs

    def get_windown_obs_in_dim(self, pointers, dim):
        signal = self.get_signal(dim)
        windown_obs = [signal[i] for i in pointers]
        return windown_obs

    def get_windown_timestamps(self, pointers):
        """
        Args:
            pointers (list of int): pointers to the trip, indicating the place from which we want to take the
            observations.

        Returns:
            windown_timestamps (list float):
        """
        timestamps = self.timestamps
        windown_ts = [timestamps[i] for i in pointers]
        return windown_ts

    def get_obs_label(self, pointer, index=0):
        """
        Args:
            pointer (int): pointer to the trip, indicating the place where the observation is.
            index (int): index of the label one wishes to extract.

        Returns:
            obs_label (any): the label of the observation in the original trip.
        """
        if self.labels is None:
            raise AttributeError("No labels were provided for this Trip")
        else:
            label_list = self.labels[index]
            obs_label = label_list[pointer]
            return obs_label

    def get_window_labels(self, pointers, index=0):
        """
        Args:
            pointers: pointer to the trip, indicating the place where all windows' observations are.
            index (int): index of the label one wishes to extract.

        Returns:
            windown_labels (list): list of labels of the window's observations in the original trip. Each entry of the
            list corresponds to a single label of the observation taken from the trip's time-series and thus, the length
            of the list is equal to the length of the pointers' list.
        """
        if self.labels is None:
            raise AttributeError("No labels were provided for this Trip")
        else:
            windown_labels = []
            for pointer in pointers:
                point_label = self.get_obs_label(pointer, index)
                windown_labels.append(point_label)
            return windown_labels

    def get_signal(self, dim):
        """
        Args:
            dim (int): index of signal that should be extracted.
        Returns:
            signal (numpy.array): time-series of the trip's signal.
        """
        signal = self.signals[dim]
        return signal

    def update_signal(self, dim, new_signal):
        new_signal_list = self.signals
        new_signal_list[dim] = new_signal
        self.signals = new_signal_list

    def get_trip_size(self):
        """
        Returns:
            n_obs (int): length of the signals list, i.e. the number of measurements of the trip.
        """
        return self.n_obs

    def get_trip_dimensions(self):
        """
        Returns:
            n_dim (int): dimension of the trip (i.e. number fo signals)
        """
        return self.n_dim

    def get_timestamps(self):
        """
        Returns:
            timestamps (list of float): list wit the timestamp of each observation
        """
        return self.timestamps

    def get_label_pointers(self, label_index, ignore_label=None):
        labels_array = np.array(self.labels[label_index])
        labels_set = set(labels_array)
        if ignore_label is not None:
            labels_set = labels_set.difference({ignore_label})
        labels_dict = dict()
        for label in labels_set:
            labels_dict[label] = np.argwhere(labels_array == label).reshape(-1).tolist()
        return labels_dict
