import time
from tripMD.vsax.objects.vsax_sequence import VSaxSequence
from tripMD.objects.motif import Motif


def find_all_motifs_in_trip_list(
    trip_list, default_letter_size, min_word_size, max_radius, compute_mdl=False,
):
    vsax_sequence = VSaxSequence(trip_list, default_letter_size)
    motif_list = []
    word_size = min_word_size
    while True:
        start_time = time.time()
        word_list = vsax_sequence.get_word_list(word_size)
        word_str_list = [word.get_str_rep() for word in word_list]
        word_patterns = set(word_str_list)
        # if there are no repeating patterns of size word_size, then there are no more motifs and the loop ends
        if len(word_str_list) == len(word_patterns):
            break
        else:
            for pattern in word_patterns:
                # if the pattern appears only once, then it is not a motif
                if word_str_list.count(pattern) < 2:
                    continue
                else:
                    pattern_motif = Motif(pattern, max_radius, compute_mdl)
                    pattern_motif.compute_center_and_members(vsax_sequence, word_list)
                    # if the pattern only has one member, then it is not a motif
                    if len(pattern_motif.get_members()) == 1:
                        continue
                    else:
                        motif_list.append(pattern_motif)
        print_time(word_size, start_time)
        word_size = word_size + 1
    return motif_list


def print_time(word_size, start_time):
    time_diff = time.time() - start_time
    if time_diff > 3600:
        print(
            "All motifs of size {} successfully extracted in {} hours".format(
                word_size, round(time_diff / 3600, 2)
            )
        )
    elif time_diff > 60:
        print(
            "All motifs of size {} successfully extracted in {} minutes".format(
                word_size, round(time_diff / 60, 2)
            )
        )
    else:
        print(
            "All motifs of size {} successfully extracted in {} seconds".format(
                word_size, round(time_diff, 2)
            )
        )
