import numpy as np
from tripMD.objects.trip import Trip


def load_trips_from_folder(folder_path):
    # todo: implement load_trips_from_folder
    pass


def load_trips_from_dataframe(
    data_df,
    trip_id_field,
    excluded_fields=None,
    timestamp_field=None,
    label_fields=None,
):
    if excluded_fields is None:
        excluded_fields = {trip_id_field}
    else:
        excluded_fields = excluded_fields.union({trip_id_field})
    if label_fields is not None:
        excluded_fields = excluded_fields.union(set(label_fields))
    if timestamp_field is not None:
        excluded_fields = excluded_fields.union({timestamp_field})
    signal_columns_names = sorted(set(data_df.columns).difference(excluded_fields))
    unique_trip_ids = sorted(set(data_df[trip_id_field]))
    trip_list = []
    for trip_id in unique_trip_ids:
        trip_df = data_df[data_df[trip_id_field] == trip_id].drop(trip_id_field, axis=1)
        signal_list = [np.array(trip_df[col_name]) for col_name in signal_columns_names]
        if label_fields is None:
            if timestamp_field is None:
                trip = Trip(signal_list, trip_id)
            else:
                timestamps = trip_df[timestamp_field].tolist()
                trip = Trip(signal_list, trip_id, timestamps=timestamps)
        else:
            label_list = [trip_df[col_name].tolist() for col_name in label_fields]
            if timestamp_field is None:
                trip = Trip(signal_list, trip_id, labels_list=label_list)
            else:
                timestamps = trip_df[timestamp_field].tolist()
                trip = Trip(
                    signal_list, trip_id, timestamps=timestamps, labels_list=label_list
                )
        trip_list.append(trip)
    return trip_list
