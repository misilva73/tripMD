import math
from tripMD.dtwdist import compute_ndim_dwt_dist_between_ts_and_list


def compute_mdl_cost(members_word_list, trip_word_list):
    """
    This function compute the MDL cost associated to the motif that generated the members in members_word_list,
    as proposed by Tanaka et all.

    Args:
        members_word_list (list of tripMD.vsax.objects.vsax_word.VSaxWord): List of all the vsax words that make up the
        motif.
        trip_word_list (list of tripMD.vsax.objects.vsax_word.VSaxWord): List of all words in extracted from the original
        trip.

    Returns:
        mdl_cost (float): MDL cost of the given motif.
    """
    split_word_lengths_list = _compute_split_word_lengths_list(
        members_word_list, trip_word_list
    )
    mdl_cost = _compute_mdl_cost_of_split_list(split_word_lengths_list)
    return mdl_cost


def prune_motifs(trip_list, motif_list, max_radius):
    """
    This function prunes the set of all extracted motifs based on the MDL cost and the distance of the centers. In short,
    motifs with the highest MDL cost get successively added to the pruned list if the distance of their center to the
    centers of all the previously selected motifs is higher than 2*max_radius.

    Args:
        trip_list (list of tripMD.objects.trip.Trip): original list of Trips.
        motif_list (list of tripMD.objects.motif.Motif): list of all the motifs extracted from the trip_list.
        max_radius (float): maximum radius to define the motifs.

    Returns:
        pruned_motif_list (list of tripMD.objects.motif.Motif): list of pruned motifs.

    """
    sorted_motif_list = sorted(motif_list, key=lambda x: x.get_mdl_cost())
    pruned_motif_list = [sorted_motif_list[0]]
    first_center_obs = sorted_motif_list[0].get_center_obs(trip_list)
    pruned_center_obs_list = [first_center_obs]
    for motif in sorted_motif_list[1:]:
        current_center_obs = motif.get_center_obs(trip_list)
        dist_list = compute_ndim_dwt_dist_between_ts_and_list(
            current_center_obs, pruned_center_obs_list, 2 * max_radius
        )
        dist_test_list = [dist <= 2 * max_radius for dist in dist_list]
        if sum(dist_test_list) == 0:
            pruned_motif_list.append(motif)
            pruned_center_obs_list.append(current_center_obs)
        else:
            continue
    return pruned_motif_list


def _compute_split_word_lengths_list(members_word_list, trip_word_list):
    """
    This function computes the split list with the segments of lengths from trip_word_list, based on the position of the
    motif's members provided in members_word_list. This is the first step in the MDL cost computation proposed by
    Tanaka et all.

    Args:
        members_word_list (list of tripMD.vsax.objects.vsax_word.VSaxWord): List of all the vsax words that make up the
        motif.
        trip_word_list (list of tripMD.vsax.objects.vsax_word.VSaxWord): List of all words in extracted from the original
        trip.

    Returns:
        split_word_lengths_list (list os list of int): list with the split segments of the lengths.
    """
    split_word_lengths_list = []
    trip_word_lengths = [word.get_word_lengths() for word in trip_word_list]
    previous_index = 0
    for member in members_word_list:
        member_index = _find_word_index_in_list(member, trip_word_list)
        if member_index > 0:
            next_segment_lengths = sum(
                trip_word_lengths[previous_index:member_index], []
            )
            split_word_lengths_list.append(next_segment_lengths)
        member_length = trip_word_lengths[member_index]
        split_word_lengths_list.append(member_length)
        previous_index = member_index
    if previous_index < len(trip_word_lengths):
        next_segment_lengths = sum(trip_word_lengths[previous_index:], [])
        split_word_lengths_list.append(next_segment_lengths)
    return split_word_lengths_list


def _find_word_index_in_list(word, word_list):
    word_trip_id = word.get_trip_index()
    word_pointers = word.get_pointers()
    for index, candidate_word in enumerate(word_list):
        candidate_trip_id = candidate_word.get_trip_index()
        if candidate_trip_id == word_trip_id:
            candidate_pointers = candidate_word.get_pointers()
            if candidate_pointers == word_pointers:
                return index
    raise IndexError("The word provided is not in the word_list")


def _compute_mdl_cost_of_split_list(split_word_lengths_list):
    """
    This function computes the MDL cost associated with the list of split word lengths. This is a final step in the MDL
    cost computation proposed by Tanaka et all.

    Args:
        split_word_lengths_list (list os list of int): list with the split segments of the lengths.

    Returns:
        mdl_cost (float): MDL cost of the given list
    """
    par_cost_list = []
    data_cost_list = []
    for split in split_word_lengths_list:
        split_len_sum = float(sum(split))
        split_par_cost = math.log2(split_len_sum)
        par_cost_list.append(split_par_cost)
        split_data_cost_list = [-l * math.log2(l / split_len_sum) for l in split]
        split_data_cost = sum(split_data_cost_list)
        data_cost_list.append(split_data_cost)
    par_cost = sum(par_cost_list)
    data_cost = sum(data_cost_list)
    split_cost = len(split_word_lengths_list) * math.log2(
        sum(sum(split_word_lengths_list, []))
    )
    mdl_cost = round(par_cost + data_cost + split_cost, 2)
    return mdl_cost
