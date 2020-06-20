import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from utils.uah_eval import get_labels_counts_from_clusters


def plot_motif_in_trips(motif, trip_list, lon_index, lat_index, add_suptitle):
    """

    Args:
        motif (tripMD.objects.motif.Motif): motif to plot
        trip_list (list of tripMD.objects.trip.Trip): list of original trips from which the motif was extracted
        lon_index:
        lat_index:
        add_suptitle (bool): indicate whether to add a suptitle with the motif's pattern

    Returns:

    """
    n_rows = len(trip_list)
    fig, axs = plt.subplots(
        nrows=n_rows, ncols=2, figsize=(15, n_rows * 2), sharex=True, sharey=True
    )
    if add_suptitle:
        plt.suptitle(str(motif.get_description()), y=1.05)
    # Plot full trips
    for i, trip in enumerate(trip_list):
        timestamps = trip.get_timestamps()
        axs[i, 0].plot(timestamps, trip.get_signal(lat_index), "xkcd:grey", alpha=0.5)
        axs[i, 0].set_title("Trip {} : Lateral".format(str(i)))
        axs[i, 1].plot(timestamps, trip.get_signal(lon_index), "xkcd:grey", alpha=0.5)
        axs[i, 1].set_title("Trip {} : Longitudinal".format(str(i)))
    # Add members to plots
    for member in motif.get_members():
        trip_id = member.get_trip_index()
        trip = trip_list[trip_id]
        pointers = member.get_pointers()
        timestamps = trip.get_windown_timestamps(pointers)
        axs[trip_id, 0].plot(
            timestamps,
            trip.get_windown_obs_in_dim(pointers, lat_index),
            "xkcd:dark grey",
        )
        axs[trip_id, 1].plot(
            timestamps,
            trip.get_windown_obs_in_dim(pointers, lon_index),
            "xkcd:dark grey",
        )
    plt.tight_layout()
    return fig


def plot_motif_members(motif, trip_list, lon_index, lat_index, add_suptitle):
    fig, axs = plt.subplots(ncols=2, figsize=(8, 2))
    if add_suptitle:
        plt.suptitle(str(motif.get_description()), y=1.05)
    # Plot members
    for member in motif.get_members():
        trip_id = member.get_trip_index()
        trip = trip_list[trip_id]
        pointers = member.get_pointers()
        timestamps = np.arange(len(pointers))
        axs[0].plot(
            timestamps,
            trip.get_windown_obs_in_dim(pointers, lat_index),
            "xkcd:dark grey",
        )
        axs[0].set_title("Lateral Acceleration")
        axs[1].plot(
            timestamps,
            trip.get_windown_obs_in_dim(pointers, lon_index),
            "xkcd:dark grey",
        )
        axs[1].set_title("Longitudinal Acceleration")
    # Plot center
    center = motif.get_center()
    trip_id = center.get_trip_index()
    trip = trip_list[trip_id]
    pointers = center.get_pointers()
    timestamps = np.arange(len(pointers))
    axs[0].plot(
        timestamps, trip.get_windown_obs_in_dim(pointers, lat_index), "xkcd:azure"
    )
    axs[1].plot(
        timestamps, trip.get_windown_obs_in_dim(pointers, lon_index), "xkcd:azure"
    )
    # Add zero line
    axs[0].axhline(y=0, color="tomato", alpha=0.4)
    axs[1].axhline(y=0, color="tomato", alpha=0.4)
    # Fix plot axis
    lat_min, lat_max, lon_min, lon_max = compute_trips_max_min(
        trip_list, lon_index, lat_index
    )
    axs[0].set_ylim(lat_min, lat_max)
    axs[1].set_ylim(lon_min, lon_max)
    plt.tight_layout()
    return fig


def plot_clusters_acceleration(
    dtwsom_model,
    motif_center_list,
    freq_per_second,
    acc_index,
    color="midnightblue",
    suptitle="Acceleration",
    figsize=(10, 6),
):
    n_neurons = dtwsom_model._size
    n = dtwsom_model._cols
    fig, axs = plt.subplots(n, n, figsize=figsize, sharex=True, sharey=True)
    for neuron_index in range(n_neurons):
        col = math.floor(neuron_index / n)
        row = neuron_index % n
        cluster_list = dtwsom_model.capture_objects[neuron_index]
        for member_index in cluster_list:
            acc_obs = [obs[acc_index] for obs in motif_center_list[member_index]]
            axs[row, col].plot(
                np.round(np.arange(len(acc_obs)) / freq_per_second, 2),
                acc_obs,
                alpha=0.3,
                color=color,
            )
        axs[row, col].set_title("Cluster: " + str(neuron_index))
        if col == 0:
            axs[row, col].set_ylabel("Seq. values")
        if row == n - 1:
            axs[row, col].set_xlabel("Time in seconds")
    plt.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def compute_trips_max_min(trip_list, lon_index, lat_index):
    lat_min = np.inf
    lat_max = -np.inf
    lon_min = np.inf
    lon_max = -np.inf
    for trip in trip_list:
        lat_min, lat_max = update_min_max(lat_min, lat_max, trip, lat_index)
        lon_min, lon_max = update_min_max(lon_min, lon_max, trip, lon_index)
    return lat_min, lat_max, lon_min, lon_max


def update_min_max(curr_min, curr_max, trip, index):
    trip_signal = trip.get_signal(index)
    trip_max = max(trip_signal)
    if trip_max > curr_max:
        new_max = trip_max
    else:
        new_max = curr_max
    trip_min = min(trip_signal)
    if trip_min < curr_min:
        new_min = trip_min
    else:
        new_min = curr_min
    return new_min, new_max


def plot_cluster_behavior_rates(
    cluster_dict_list, trip_list, labeled_trips_ids, n_cols
):
    label_counts_df = get_labels_counts_from_clusters(
        cluster_dict_list, trip_list, labeled_trips_ids
    )
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 3.2), constrained_layout=True)
    for n, behavior in enumerate(["normal", "aggressive", "drowsy"]):
        rate_series = label_counts_df[
            label_counts_df["behavior"] == behavior
        ].set_index("cluster_id")["label_rates"]
        rate_mat = np.zeros((n_cols, n_cols))
        for i in range(n_cols):
            for j in range(n_cols):
                neuron_index = i * n_cols + j
                axs[n].text(i, j, str(neuron_index), va="center", ha="center")
                if neuron_index in rate_series:
                    rate_mat[i][j] = rate_series[neuron_index]
                else:
                    rate_mat[i][j] = 0
        im = axs[n].imshow(rate_mat, cmap=plt.get_cmap("YlOrBr"), vmin=0, vmax=1)
        axs[n].axes.get_xaxis().set_visible(False)
        axs[n].axes.get_yaxis().set_visible(False)
        axs[n].set_title(behavior + " behavior")
    fig.colorbar(im, ax=axs, shrink=0.6, orientation="horizontal")
    return fig
