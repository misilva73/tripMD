import os
import sys
import fire
import time
import pickle
from pathlib import Path
import random
import numpy as np

random.seed(0)
np.random.seed(0)

# Internal package imports
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
from tripMD import load, main
from utils import uah_data

DEFAULT_DATA_PATH = os.path.abspath(os.path.join(ROOT_DIR, os.pardir, "data-uah"))
DEFAULT_DRIVER_PATH = os.path.abspath(os.path.join(ROOT_DIR, os.pardir, "data-uah/D2"))


def run_all_drivers(data_path=DEFAULT_DATA_PATH, freq_per_second=5):
    output_folder = os.path.join(ROOT_DIR, "outputs/all_drivers")
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    # Load data
    abs_data_path = os.path.abspath(data_path)
    print("-- Running TripMD eval for all drivers :")
    data_df = uah_data.get_full_point_uah_data(abs_data_path, freq_per_second)
    # Filtering two trips with data errors!
    clean_data_df = data_df[
        ~data_df["trip_id"].isin(["20151126134736", "20151211160213"])
    ]
    run_tripmd_pipeline(clean_data_df, output_folder, freq_per_second)


def run_driver(
    driver_data_path=DEFAULT_DRIVER_PATH, freq_per_second=5, output_name="D2_driver"
):
    output_folder = os.path.join(ROOT_DIR, "outputs", output_name)
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    # Load data
    abs_data_path = os.path.abspath(driver_data_path)
    print("-- Running TripMD eval for {} :".format(abs_data_path))
    data_df = uah_data.get_full_point_uah_data(abs_data_path, freq_per_second)
    # Filtering two trips with data errors!
    clean_data_df = data_df[
        ~data_df["trip_id"].isin(["20151126134736", "20151211160213"])
    ]
    run_tripmd_pipeline(clean_data_df, output_folder, freq_per_second)


def run_tripmd_pipeline(data_df, output_folder, freq_per_second):
    start_time = time.time()
    # Create trip list
    trip_id_field = "trip_id"
    excluded_fields = {"ax", "event_level", "event", "user_id", "road"}
    label_fields = ["lc_event", "event_type", "trip_label"]
    timestamp_field = "timestamp"
    trip_list = load.load_trips_from_dataframe(
        data_df,
        trip_id_field,
        excluded_fields=excluded_fields,
        timestamp_field=timestamp_field,
        label_fields=label_fields,
    )
    save_file = os.path.join(output_folder, "trip_list.p")
    pickle.dump(trip_list, open(save_file, "wb"))
    print("Data was loaded and trip_list was created")
    # Init tripMD
    trip_md = main.TripMD(trip_list, freq_per_second, estimate_max_radius=True)
    print(
        "Running TripMD with a max radius of {}".format(round(trip_md._max_radius, 4))
    )
    trip_md.set_parameters(
        lat_acc_index=0, lon_acc_index=1, output_folder=output_folder
    )
    # Run tripMD main functions
    motif_list = trip_md.run_motif_extraction()
    motif_list = trip_md.run_maneuver_description(motif_list)
    pruned_motif_list = trip_md.run_motif_pruning(motif_list)
    cluster_dict_list = trip_md.run_dtw_som(motif_list, pruned_motif_list)
    _ = trip_md.compute_clusters_summary(cluster_dict_list)
    # Print time
    print_time(start_time)


def print_time(start_time):
    time_diff = time.time() - start_time
    if time_diff > 3600:
        print(
            "TripMD eval pipeline was run in {} hours".format(
                round(time_diff / 3600, 2)
            )
        )
    elif time_diff > 60:
        print(
            "TripMD eval pipeline was run in {} minutes".format(
                round(time_diff / 60, 2)
            )
        )
    else:
        print("TripMD eval pipeline was run in {} seconds".format(round(time_diff, 2)))


if __name__ == "__main__":
    fire.Fire(
        {"all_drivers": run_all_drivers, "single_driver": run_driver,}
    )
