import numpy as np


def compute_vsax_cuts(trip_list):
    """
    This function computes the cuts for transforming each signal into a sequence of vsax letters. The function will return
    an array of cuts for each trips' dimension and the cuts are computed consistently for all trips. This means that
    two points with the same value in two different trips will be mapped to the same letter.

    Args:
        trip_list (list of tripMD.objects.trip.Trip): original list of trips that need to be normalized

    Returns:
        cuts_dict (dict of numpy.array): dictionary where the keys are the trips' dimensions and the values are that
        dimensions' vsax cuts

    """
    n_dim = get_n_dim_from_trip_list(trip_list)
    cuts_dict = {}
    for dim in range(n_dim):
        signal_list = get_1dim_signal_list(trip_list, dim)
        cuts_dict[dim] = get_1dim_cuts(signal_list)
    return cuts_dict


def get_n_dim_from_trip_list(trip_list):
    """

    Args:
        trip_list (list of tripMD.objects.Trip): original list of trips that need to be normalized

    Returns:

    """
    n_dim_list = [trip.get_trip_dimensions() for trip in trip_list]
    if len(set(n_dim_list)) != 1:
        raise Exception(
            "Dimension of provided trips is inconsistent. Make sure that all trips"
            " have the same feature/signals"
        )
    else:
        return n_dim_list[0]


def get_1dim_signal_list(trip_list, dim):
    """

    Args:
        trip_list (list of tripMD.objects.Trip): original list of trips that need to be normalized
        dim: position of the signal. Each Trip object has a list of signals, where each signal corresponds to a
        specific feature/measurement. The dim

    Returns:

    """
    signal_list = []
    for trip in trip_list:
        signal = trip.get_signal(dim)
        signal_list.append(signal)
    return signal_list


def get_1dim_cuts(signal_list):
    """

    Args:
        signal_list (list of numpy.array): list of 1-dim signals, taken from the list of trips (the signal in the n-th
        position corresponds to the n-th trip in the trip_list)

    Returns:

    """
    joined_signal = np.concatenate(signal_list)
    centered_signal = joined_signal - np.mean(joined_signal)
    p5 = np.percentile(centered_signal, 5)
    p15 = np.percentile(centered_signal, 15)
    p85 = np.percentile(centered_signal, 85)
    p95 = np.percentile(centered_signal, 95)
    signal_cuts = np.array([-np.inf, p5, p15, p85, p95])
    return signal_cuts
