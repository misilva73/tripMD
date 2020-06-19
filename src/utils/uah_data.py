import os
import pandas as pd
import numpy as np


def get_full_point_uah_data(data_path, freq_per_second=10):
    # initialize data variable as list
    data_list = []
    for root, dirs, files in os.walk(data_path, topdown=False):
        if len(files) < 2:
            continue
        else:
            # import individual trip files
            inertial_df = import_uah_inertial(root)
            events_df = import_uah_events(root)
            lc_df = import_uah_lc(root)
            # transform and join trip files
            trip_df = transform_uah_trip_data(
                inertial_df, events_df, lc_df, freq_per_second
            )
            # add ids and labels to trip_df
            trip_df["user_id"] = root.split("/")[-1].split("-")[2]
            trip_df["trip_id"] = root.split("/")[-1].split("-")[0]
            trip_df["trip_label"] = get_trip_labels(root, len(trip_df.index))
            trip_df["road"] = get_road_type(root)
            # append trip to data_list
            data_list.append(trip_df)
    return pd.concat(data_list, ignore_index=True, sort=False)


def import_uah_inertial(root_path):
    file_path = os.path.join(root_path, "RAW_ACCELEROMETERS.txt")
    inertial_df = pd.read_csv(file_path, sep=" ", header=None).iloc[:, 0:11]
    # add the column names
    inertial_df.columns = [
        "timestamp",
        "activated",
        "raw_ax",
        "raw_ay",
        "raw_az",
        "ax",
        "ay",
        "az",
        "roll",
        "pitch",
        "yaw",
    ]
    return inertial_df


def import_uah_events(root_path):
    file_path = os.path.join(root_path, "EVENTS_INERTIAL.txt")
    try:
        events_df = pd.read_csv(file_path, sep=" ", header=None).iloc[:, 0:6]
        # add the column names
        events_df.columns = [
            "timestamp",
            "event_type",
            "event_level",
            "latitude",
            "longitude",
            "date",
        ]
    except:
        events_df = pd.DataFrame(
            columns=["timestamp", "event_type", "event_level", "latitude", "longitude", "date"]
        )
    return events_df


def import_uah_lc(root_path):
    file_path = os.path.join(root_path, "EVENTS_LIST_LANE_CHANGES.txt")
    try:
        lc_df = pd.read_csv(file_path, sep=" ", header=None)
        # add the column names
        lc_df.columns = ["timestamp", "lc_event", "lat", "lon", "duration", "threshold"]
        lc_len = len(lc_df.index)
        for i in range(lc_len):
            start = lc_df.loc[i]["timestamp"] + 0.1
            end = lc_df.loc[i]["timestamp"] + lc_df.loc[i]["duration"] + 0.1
            for time in np.arange(start, end, 0.1):
                new_row = {
                    "timestamp": time,
                    "lc_event": lc_df.loc[i]["lc_event"],
                }
                lc_df = lc_df.append(new_row, ignore_index=True)
        lc_df = lc_df[["timestamp", "lc_event"]]
    except:
        lc_df = pd.DataFrame(columns=["timestamp", "lc_event"])
    return lc_df


def transform_uah_trip_data(inertial_df, events_df, lc_df, freq_per_second):
    trans_inertial_df = (
        inertial_df.drop(
            columns=["activated", "raw_ax", "raw_ay", "raw_az", "roll", "pitch", "yaw"]
        )
        .assign(
            timestamp=lambda x: np.round(x.timestamp * freq_per_second)
            / freq_per_second
        )
        .groupby("timestamp")
        .agg("mean")
        .reset_index()
    )
    trans_events_df = (
        events_df.assign(
            timestamp=lambda x: np.round(x.timestamp * freq_per_second)
            / freq_per_second
        )
        .set_index("timestamp")
        .drop(columns=["latitude", "longitude", "date"])
    )
    trans_lc_df = lc_df.assign(
        timestamp=lambda x: np.round(x.timestamp * freq_per_second) / freq_per_second
    ).set_index("timestamp")
    final_df = (
        trans_inertial_df.join(trans_events_df, on="timestamp")
        .join(trans_lc_df, on="timestamp")
        .fillna(0)
        .assign(
            event=lambda x: x["event_type"].astype(int).astype(str)
            + x["event_level"].astype(int).astype(str)
        )
    )
    return final_df


def get_trip_labels(file_path, array_size):
    if "NORMAL" in file_path:
        label = "normal"
    elif "AGGRESSIVE" in file_path:
        label = "aggressive"
    elif "DROWSY" in file_path:
        label = "drowsy"
    else:
        label = None
    return np.repeat(label, array_size)


def get_road_type(file_path):
    if "SECONDARY" in file_path:
        return "secondary"
    elif "MOTORWAY" in file_path:
        return "motorway"
    else:
        return None
