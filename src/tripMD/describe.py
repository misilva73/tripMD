LAT_LETTER_DICT = {
    "high_neg": "high_neg_turn",
    "low_neg": "neg_turn",
    "low_pos": "pos_turn",
    "high_pos": "high_pos_turn",
}

LON_LETTER_DICT = {
    "high_neg": "strong_brake",
    "low_neg": "brake",
    "low_pos": "accel",
    "high_pos": "strong_accel",
}


def describe_trip_motif(motif, lat_index, lon_index):
    pattern_tuple = motif.get_pattern()
    lat_word = pattern_tuple[lat_index]
    lon_word = pattern_tuple[lon_index]
    maneuvers_dict = _describe_from_sax_words(lat_word, lon_word)
    return maneuvers_dict


def _describe_from_sax_words(lat_word, lon_word):
    lat_maneuvers = _split_word_in_simple_maneuvers(lat_word, LAT_LETTER_DICT)
    lon_maneuvers = _split_word_in_simple_maneuvers(lon_word, LON_LETTER_DICT)
    maneuvers_dict = {"lat": lat_maneuvers, "lon": lon_maneuvers}
    return maneuvers_dict


def _split_word_in_simple_maneuvers(sax_word, acc_dict):
    maneuver_list = []
    current_maneuver = sax_word[0]
    for letter in sax_word[1:]:
        if (letter in ["a", "b", "c"]) & (current_maneuver[-1] in ["d", "e"]):
            if "e" in current_maneuver:
                maneuver_list.append(acc_dict["high_pos"])
            else:
                maneuver_list.append(acc_dict["low_pos"])
            current_maneuver = letter
        elif (letter in ["d", "e", "c"]) & (current_maneuver[-1] in ["a", "b"]):
            if "a" in current_maneuver:
                maneuver_list.append(acc_dict["high_neg"])
            else:
                maneuver_list.append(acc_dict["low_neg"])
            current_maneuver = letter
        else:
            current_maneuver += letter
    # Adding the last maneuver if needed (i.e., if the last letter is not "c)
    if sax_word[-1] != "c":
        if "e" in current_maneuver:
            maneuver_list.append(acc_dict["high_pos"])
        elif "d" in current_maneuver:
            maneuver_list.append(acc_dict["low_pos"])
        elif "a" in current_maneuver:
            maneuver_list.append(acc_dict["high_neg"])
        else:
            maneuver_list.append(acc_dict["low_neg"])
    # check if on maneuvers were found
    if len(maneuver_list) == 0:
        maneuver_list = ["no_man"]
    return maneuver_list
