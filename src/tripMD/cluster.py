import os
import math
import dtwsom
import numpy as np
import matplotlib.pyplot as plt
from pyclustering.nnet.som import type_conn


def fit_dtwsom_on_motifs(
    motif_list, pruned_motif_list, trip_list, epochs, dtw_window_size
):
    motif_centers = [motif.get_center_obs(trip_list) for motif in motif_list]
    pruned_motif_centers = [
        motif.get_center_obs(trip_list) for motif in pruned_motif_list
    ]
    dtwsom_model = _init_dtwsom_network(pruned_motif_centers, dtw_window_size)
    dtwsom_model.train(motif_centers, epochs, anchors=pruned_motif_centers)
    return dtwsom_model


def get_motif_clusters(dtwsom_model, motif_list):
    cluster_dict_list = []
    dtwsom_units_list = dtwsom_model.weights
    dtwsom_clusters_list = dtwsom_model.capture_objects
    for i in range(len(dtwsom_units_list)):
        cluster_members = dtwsom_clusters_list[i]
        cluster_dict = {
            "cluster_ts": dtwsom_units_list[i],
            "cluster_members": [motif_list[j] for j in cluster_members],
        }
        cluster_dict_list.append(cluster_dict)
    return cluster_dict_list


def plot_and_save_dtwsom_network(dtwsom_model, output_folder):
    save_file = os.path.join(output_folder, "umatrix_plot.png")
    dtwsom_model.save_distance_matrix(save_file)
    plt.close()
    plt.clf()
    save_file = os.path.join(output_folder, "winner_plot.png")
    dtwsom_model.save_winner_matrix(save_file)
    plt.close()
    plt.clf()


def plot_dtwsom_bmus(
    dtwsom_model, lat_index, lon_index, freq_per_second, suptitle=None, figsize=(10, 6)
):
    n_neurons = dtwsom_model.size
    fig, axs = plt.subplots(
        dtwsom_model._rows,
        dtwsom_model._cols,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )
    for neuron_index in range(n_neurons):
        n = dtwsom_model._rows
        col = math.floor(neuron_index / n)
        row = neuron_index % n
        neuron_lat_ys = [
            weight[lat_index] for weight in dtwsom_model._weights[neuron_index]
        ]
        neuron_lon_ys = [
            weight[lon_index] for weight in dtwsom_model._weights[neuron_index]
        ]
        xs = np.round(np.arange(len(neuron_lat_ys)) / freq_per_second, 2)
        axs[row, col].axhline(y=0, color='k', alpha=0.5, linestyle='-')
        axs[row, col].plot(
            xs, neuron_lat_ys, color="midnightblue", label="Lateral Acceleration",
        )
        axs[row, col].plot(
            xs, neuron_lon_ys, color="darkorange", label="Longitudinal Acceleration",
        )
        axs[row, col].set_title("Unit: " + str(neuron_index))
        if col == 0:
            axs[row, col].set_ylabel("Seq. values")
        if row == dtwsom_model._rows - 1:
            axs[row, col].set_xlabel("Time in seconds")
    if suptitle is not None:
        plt.suptitle(suptitle)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        plt.tight_layout()
    plt.legend(bbox_to_anchor=(-1.8, -0.7), loc=2, borderaxespad=0.0)
    return fig


def plot_and_save_dtwsom_bmus(
    dtwsom_model, lat_index, lon_index, freq_per_second, output_folder
):
    fig = plot_dtwsom_bmus(dtwsom_model, lat_index, lon_index, freq_per_second)
    save_file = os.path.join(output_folder, "bmus_plot.png")
    fig.savefig(save_file)
    plt.close(fig)


def _init_dtwsom_network(pruned_motif_centers, dtw_window_size):
    rows, cols, structure = _define_network_structure(pruned_motif_centers)
    parameters = dtwsom.DtwSomParameters()
    parameters.init_type = dtwsom.DtwTypeInit.anchors
    dtw_params = dtwsom.DtwParameters(window=dtw_window_size)
    network = dtwsom.DtwSom(
        rows, cols, structure, parameters=parameters, dtw_params=dtw_params
    )
    return network


def _define_network_structure(pruned_motif_centers):
    n_anchors = len(pruned_motif_centers)
    if n_anchors <= 4:
        rows = 2
        cols = 2
    else:
        sqrt_anchors = math.ceil(math.sqrt(n_anchors))
        rows = sqrt_anchors
        cols = sqrt_anchors
    structure = type_conn.grid_four
    return rows, cols, structure
