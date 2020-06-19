import os
import time
import pickle
import random
import numpy as np
from pathlib import Path
from dtaidistance import dtw_ndim
from tripMD.describe import describe_trip_motif
from tripMD.find_motifs import find_all_motifs_in_trip_list
from tripMD.mdl import prune_motifs
from tripMD.cluster import (
    fit_dtwsom_on_motifs,
    get_motif_clusters,
    plot_and_save_dtwsom_network,
    plot_and_save_dtwsom_bmus,
)


class TripMD(object):
    def __init__(self, trip_list, freq_per_second, estimate_max_radius=False):
        self._trip_list = trip_list
        self._default_letter_size = freq_per_second  # 1 second letters
        self._min_word_size = 3
        if estimate_max_radius:
            self._max_radius = self.estimate_max_radius()
        else:
            self._max_radius = None
        self._compute_mdl = True
        self._lat_acc_index = None
        self._lon_acc_index = None
        self._dtw_som_epochs = 20
        self._output_folder = os.path.join(
            os.getcwd(), "outputs", str(round(time.time()))
        )
        self._freq_per_second = freq_per_second
        self._dtwsom_window_size = int(
            round(freq_per_second / 2)
        )  # half a second max shifts for DTW-SOM

    def set_parameters(self, **kwargs):
        for k, v in kwargs.items():
            if k == "default_letter_size":
                self._default_letter_size = v
            elif k == "min_word_size":
                self._min_word_size = v
            elif k == "max_radius":
                self._max_radius = v
            elif k == "lat_acc_index":
                self._lat_acc_index = v
            elif k == "lon_acc_index":
                self._lon_acc_index = v
            elif k == "dtw_som_epochs":
                self._dtw_som_epochs = v
            elif k == "output_folder":
                self._output_folder = v
            else:
                raise Warning("The provided key {} a user-defined parameter".format(k))

    def estimate_max_radius(self):
        sample_obs_list = self.sample_trip_list(2000)
        dist_matrix = dtw_ndim.distance_matrix(sample_obs_list, window=1)
        dist_lst = np.extract(dist_matrix < np.inf, dist_matrix)
        max_radius = np.percentile(dist_lst, 0.5)
        return max_radius

    def sample_trip_list(self, sample_size):
        window_size = self._default_letter_size * self._min_word_size
        sample_obs_list = []
        for i in range(sample_size):
            sample_trip = random.choice(self._trip_list)
            sample_trip_size = sample_trip.get_trip_size()
            sample_pointer = random.randrange(0, sample_trip_size - window_size)
            sample_window = list(range(sample_pointer, sample_pointer + window_size))
            sample_obs = sample_trip.get_windown_obs(sample_window)
            sample_obs_list.append(sample_obs)
        return sample_obs_list

    def run_pipeline(self, checkpoint=True):
        motif_list = self.run_motif_extraction(checkpoint)
        motif_list = self.run_maneuver_description(motif_list, checkpoint)
        pruned_motif_list = self.run_motif_pruning(motif_list, checkpoint)
        cluster_dict_list = self.run_dtw_som(motif_list, pruned_motif_list, checkpoint)
        summary_dict_list = self.compute_clusters_summary(cluster_dict_list)
        return summary_dict_list

    def run_motif_extraction(self, checkpoint=True):
        motif_list = find_all_motifs_in_trip_list(
            self._trip_list,
            self._default_letter_size,
            self._min_word_size,
            self._max_radius,
            self._compute_mdl,
        )
        if checkpoint:
            Path(self._output_folder).mkdir(parents=True, exist_ok=True)
            save_file = os.path.join(self._output_folder, "motifs.p")
            pickle.dump(motif_list, open(save_file, "wb"))
        return motif_list

    def run_maneuver_description(self, motif_list, checkpoint=True):
        if self._lat_acc_index is None:
            raise AttributeError(
                "In order to add the maneuver description to the motifs, the user must "
                "define the lat_acc_index attribute using the set_parameters function"
            )
        elif self._lon_acc_index is None:
            raise AttributeError(
                "In order to add the maneuver description to the motifs, the user must "
                "define the lon_acc_index attribute using the set_parameters function"
            )
        else:
            for motif in motif_list:
                description_dict = describe_trip_motif(
                    motif, self._lat_acc_index, self._lon_acc_index
                )
                motif.add_description(description_dict)
            if checkpoint:
                Path(self._output_folder).mkdir(parents=True, exist_ok=True)
                save_file = os.path.join(self._output_folder, "motifs.p")
                pickle.dump(motif_list, open(save_file, "wb"))
        return motif_list

    def run_motif_pruning(self, motif_list, checkpoint=True):
        pruned_motif_list = prune_motifs(self._trip_list, motif_list, self._max_radius)
        if checkpoint:
            Path(self._output_folder).mkdir(parents=True, exist_ok=True)
            save_file = os.path.join(self._output_folder, "pruned_motifs.p")
            pickle.dump(pruned_motif_list, open(save_file, "wb"))
        return pruned_motif_list

    def run_dtw_som(
        self, motif_list, pruned_motif_list, checkpoint=True,
    ):
        dtwsom_model = fit_dtwsom_on_motifs(
            motif_list,
            pruned_motif_list,
            self._trip_list,
            self._dtw_som_epochs,
            self._dtwsom_window_size,
        )
        if checkpoint:
            # Dump model
            Path(self._output_folder).mkdir(parents=True, exist_ok=True)
            save_file = os.path.join(self._output_folder, "dtwsom_model.p")
            pickle.dump(dtwsom_model, open(save_file, "wb"))
            # Plot network
            plot_and_save_dtwsom_network(dtwsom_model, self._output_folder)
            # Plot units
            if self._lat_acc_index is None:
                raise Warning(
                    "In order to plot the DTW-SOM's BMU time-series, the user must "
                    "define the lat_acc_index attribute using the set_parameters function."
                )
            elif self._lon_acc_index is None:
                raise Warning(
                    "In order to plot the DTW-SOM's BMU time-series, the user must "
                    "define the lon_acc_index attribute using the set_parameters function."
                )
            else:
                plot_and_save_dtwsom_bmus(
                    dtwsom_model,
                    self._lat_acc_index,
                    self._lon_acc_index,
                    self._freq_per_second,
                    self._output_folder,
                )
        cluster_dict_list = get_motif_clusters(dtwsom_model, motif_list)
        if checkpoint:
            save_file = os.path.join(self._output_folder, "dtwsom_clusters.p")
            pickle.dump(cluster_dict_list, open(save_file, "wb"))
        return cluster_dict_list

    def compute_clusters_summary(self, cluster_dict_list, checkpoint=True):
        summary_dict_list = []
        for cluster_dict in cluster_dict_list:
            motif_members = cluster_dict["cluster_members"]
            n_members = len(motif_members)
            lat_description_list = [
                "-".join(motif.get_description()["lat"]) for motif in motif_members
            ]
            lon_description_list = [
                "-".join(motif.get_description()["lon"]) for motif in motif_members
            ]
            lat_values, lat_counts = np.unique(lat_description_list, return_counts=True)
            lat_maneuvers = zip(lat_values, np.round(lat_counts / n_members, 3))
            sorted_lat_maneuvers = sorted(
                lat_maneuvers, key=lambda item: item[1], reverse=True
            )
            lon_values, lon_counts = np.unique(lon_description_list, return_counts=True)
            lon_maneuvers = zip(lon_values, np.round(lon_counts / n_members, 3))
            sorted_lon_maneuvers = sorted(
                lon_maneuvers, key=lambda item: item[1], reverse=True
            )
            summary_dict = {
                "n_members": n_members,
                "lat_maneuvers": sorted_lat_maneuvers,
                "lon_maneuvers": sorted_lon_maneuvers,
            }
            summary_dict_list.append(summary_dict)
        if checkpoint:
            save_file = os.path.join(self._output_folder, "clusters_summary.p")
            pickle.dump(summary_dict_list, open(save_file, "wb"))
        return summary_dict_list
