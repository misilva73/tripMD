import random
import numpy as np
import matplotlib.pyplot as plt

from tripMD.objects.trip import Trip
from tripMD import main

random.seed(1)
np.random.seed(1)


def build_toy_trip():
    maneuvers_lat = [
        [
            -0.0175,
            -0.0175,
            -0.0285,
            -0.0285,
            -0.0515,
            -0.0515,
            -0.0375,
            -0.0525,
            -0.0645,
            -0.0705,
            -0.0360,
            -0.0225,
            -0.0305,
            -0.0175,
        ],
        [
            0.0035,
            -0.0005,
            -0.0245,
            -0.0605,
            -0.049,
            -0.040,
            -0.042,
            -0.039,
            -0.025,
            -0.007,
            -0.0045,
        ],
    ]
    maneuvers_lon = [
        [
            -0.006,
            -0.006,
            -0.013,
            -0.013,
            -0.0075,
            -0.0075,
            -0.0045,
            -0.004,
            -0.0025,
            -0.0075,
            0.002,
            0.003,
            -0.005,
            -0.0045,
        ],
        [
            -0.007,
            -0.003,
            -0.016,
            -0.013,
            -0.008,
            -0.004,
            -0.0045,
            -0.0125,
            -0.016,
            0.0,
            0.0115,
        ],
    ]

    noise_sizes = list(range(5, 10))
    final_seq_lat = []
    final_seq_lon = []

    for i in range(len(maneuvers_lat)):
        size = random.choice(noise_sizes)
        final_seq_lat += list(np.random.uniform(-0.005, 0.005, size))
        final_seq_lon += list(np.random.uniform(-0.005, 0.005, size))

        final_seq_lat += maneuvers_lat[i]
        final_seq_lon += maneuvers_lon[i]

        size = random.choice(noise_sizes)
        final_seq_lat += list(np.random.uniform(-0.005, 0.005, size))
        final_seq_lon += list(np.random.uniform(-0.005, 0.005, size))

    signal_list = [np.array(final_seq_lat), np.array(final_seq_lon)]
    trip = Trip(signal_list, 1)
    return trip


def compute_motifs_from_toy_trip(trip, **kwargs):
    trip_list = [trip]
    freq_per_second = 5

    trip_md = main.TripMD(trip_list, freq_per_second, estimate_max_radius=False)

    trip_md.set_parameters(
        lat_acc_index=0,
        lon_acc_index=1,
        output_folder=".",
    )
    if "max_radius" not in kwargs:
        trip_md.set_parameters(max_radius=0.0684)

    trip_md.set_parameters(**kwargs)

    motif_list = trip_md.run_motif_extraction(checkpoint=False)

    return motif_list


def build_trip_plot(trip):
    timestamps = trip.get_timestamps()
    trip_lat = trip.get_signal(0)
    trip_lon = trip.get_signal(1)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(timestamps, trip_lat, color="midnightblue", label="Lateral acc.")
    ax.plot(timestamps, trip_lon, color='darkorange', label="Longitudinal acc.")
    plt.ylabel("Sequence values")
    plt.xlabel("Time steps (5 steps = 1 second)")
    plt.ylim(-0.075, 0.03)
    plt.legend(loc='upper left')
    plt.tight_layout()
    return fig, ax


def build_motif_in_trip_plot(trip, motif):
    fig, ax = build_trip_plot(trip)
    for member in motif.get_members():
        pointers = member.get_pointers()
        timestamps = trip.get_windown_timestamps(pointers)
        ax.fill_between(timestamps, -0.075, 0.03, alpha=0.05, color="k")
    return fig, ax
