import numpy as np
import pandas as pd


def compute_test_driver_labels(
    cluster_dict_list, trip_list, labeled_trips_ids, driver_trips_ids
):
    all_trips_ids = sorted([trip.id for trip in trip_list])
    driver_trips_indices = np.where(np.isin(all_trips_ids, driver_trips_ids))[
        0
    ].tolist()
    labels_df = pd.DataFrame()
    for trip_index in driver_trips_indices:
        trip_labels = get_trip_predicted_behavior_label(
            trip_index, cluster_dict_list, trip_list, labeled_trips_ids
        )
        pred_label = max(trip_labels, key=trip_labels.get)
        trip_labels["trip_id"] = trip_index
        trip_labels["trip_true_label"] = trip_list[trip_index].labels[2][0]
        trip_labels["trip_pred_label"] = pred_label
        labels_df = labels_df.append(
            pd.DataFrame(trip_labels, index=[0]), ignore_index=True
        )
    labels_df.rename(
        columns={
            "aggressive": "aggressive_score",
            "normal": "normal_score",
            "drowsy": "drowsy_score",
        }
    )
    labels_df["is_pred_correct"] = (
            labels_df["trip_true_label"] == labels_df["trip_pred_label"]
    )
    return labels_df


def get_trip_predicted_behavior_label(
    trip_index, cluster_dict_list, trip_list, labeled_trips_ids
):
    label_cluster_counts_df = get_labels_counts_from_clusters(
        cluster_dict_list, trip_list, labeled_trips_ids
    )
    trip_cluster_counts_df = get_trip_cluster_counts(trip_index, cluster_dict_list)
    join_df = label_cluster_counts_df.join(
        trip_cluster_counts_df, on="cluster_id"
    ).fillna(0)
    join_df["label_score"] = join_df["label_rates"] * join_df["cluster_counts"]
    trip_predicted_labels = join_df.groupby("behavior")["label_score"].sum().to_dict()
    return trip_predicted_labels


def get_trip_cluster_counts(trip_index, cluster_dict_list):
    trip_clusters_list = []
    for cluster_id, cluster_dict in enumerate(cluster_dict_list):
        motif_list = cluster_dict["cluster_members"]
        if len(motif_list) > 0:
            for motif in motif_list:
                motif_trips = np.array(
                    [word.get_trip_index() for word in motif.members]
                )
                motif_members_in_trip = sum(motif_trips == trip_index)
                trip_clusters_list += [cluster_id] * motif_members_in_trip
    cluster, counts = np.unique(trip_clusters_list, return_counts=True)
    cluster_counts = dict(zip(cluster, counts))
    cluster_counts_df = pd.DataFrame(cluster_counts, index=[0]).T
    cluster_counts_df.columns = ["cluster_counts"]
    return cluster_counts_df


def get_labels_counts_from_clusters(cluster_dict_list, trip_list, labeled_trips_ids):
    cluster_label_counts_list = []
    for cluster_id, cluster_dict in enumerate(cluster_dict_list):
        cluster_motifs = cluster_dict["cluster_members"]
        if len(cluster_motifs) > 0:
            cluster_label_counts = get_labels_counts_from_single_cluster(
                cluster_motifs, trip_list, labeled_trips_ids
            )
            cluster_label_counts["cluster_id"] = cluster_id
            cluster_label_counts_list.append(cluster_label_counts)
    cluster_label_counts_df = pd.DataFrame(cluster_label_counts_list).fillna(0)
    final_df = cluster_label_counts_df.melt(
        id_vars=["cluster_id", "n_members", "n_labeled_members"],
        var_name="behavior",
        value_name="label_counts",
    ).sort_values("cluster_id")
    final_df["label_rates"] = final_df["label_counts"] / final_df["n_labeled_members"]
    return final_df


def get_labels_counts_from_single_cluster(
    cluster_motifs, trip_list, labeled_trips_ids=None
):
    cluster_label_counts_list = []
    for motif in cluster_motifs:
        motif_label_counts = compute_motif_behavior_counts(
            motif, trip_list, labeled_trips_ids
        )
        cluster_label_counts_list.append(motif_label_counts)
    cluster_label_counts = (
        pd.DataFrame(cluster_label_counts_list).fillna(0).sum().to_dict()
    )
    return cluster_label_counts


def compute_motif_behavior_counts(motif, trip_list, labeled_trips_ids=None):
    all_members_labels = np.array(
        [
            window_labels[0]
            for window_labels in motif.get_members_labels_list(trip_list, 2)
        ]
    )
    if labeled_trips_ids is None:
        members_labels = all_members_labels
    else:
        trip_ids = np.array([trip.id for trip in trip_list])
        labeled_trip_indexes = np.where(np.isin(trip_ids, labeled_trips_ids))[
            0
        ].tolist()
        member_trip_index = np.array(
            [word.get_trip_index() for word in motif.get_members()]
        )
        members_labels_indices = np.where(
            np.isin(member_trip_index, labeled_trip_indexes)
        )
        members_labels = all_members_labels[members_labels_indices]
    labels, counts = np.unique(members_labels, return_counts=True)
    label_counts = dict(zip(labels, counts))
    label_counts["n_members"] = len(all_members_labels)
    label_counts["n_labeled_members"] = len(members_labels)
    return label_counts
