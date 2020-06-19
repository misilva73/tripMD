from tripMD.vsax import compute_cuts
from tripMD.vsax.objects.vsax_letter import VSaxLetter
from tripMD.vsax.objects.vsax_word import VSaxWord


class VSaxSequence(object):
    """
    Object for the variable sax sequences. A variable sax sequence is a ordered list of variable sax letters computed
    from a set of trips (with multidimensional recordings).

    Args:
        trip_list (list of tripMD.objects.trip.Trip): list of trip objects with the trips that should be used to build
        the variable sax sequence.
        default_letter_size (int): minimum number of observations that build a single variable sax letter. Each letter
        will correspond to an segment of this size or bigger in the original trip recordings.

    Attributes:
        trip_list (list of tripMD.objects.trip.Trip): list of trip objects from which the variable sax sequence was built.
        n_trips (int): number of trips in the sequence
        default_letter_size (int): minimum number of observations that build a single variable sax letter. Each letter
        will correspond to an segment of this size or bigger in the original trip recordings.
        letter_seq_list (list of list): list of sequences of variable sax letters. This list includes a sequence per trip
        and, for each trip, the sequence includes that entire trip converted to a list of variable sax letters.
    """

    def __init__(self, trip_list, default_letter_size):
        """
        Constructor for the VSaxSequence class.

        Args:
            trip_list (list of tripMD.objects.trip.Trip): list of trip objects with the trips that should be used to build
            the variable sax sequence.
            default_letter_size (int): minimum number of observations that build a single variable sax letter. Each letter
            will correspond to an segment of this size or bigger in the original trip recordings.
        """
        self.trip_list = trip_list
        self.n_trips = len(trip_list)
        self.default_letter_size = default_letter_size
        self.letter_seq_list = self._compute_letter_sequence_list(
            trip_list, default_letter_size
        )

    def trip_indices(self):
        return list(range(self.n_trips))

    def trip_letters(self, trip_index):
        return self.letter_seq_list[trip_index]

    def get_trip(self, trip_index):
        return self.trip_list[trip_index]

    def get_word_list(self, word_size):
        word_list = []
        for trip_index in self.trip_indices():
            trip_word_list = self._compute_word_sequence_from_trip(
                word_size, trip_index
            )
            word_list = word_list + trip_word_list
        return word_list

    def get_word_obs(self, vsax_word):
        pointers = vsax_word.get_pointers()
        trip_index = vsax_word.get_trip_index()
        trip = self.get_trip(trip_index)
        word_obs = trip.get_windown_obs(pointers)
        return word_obs

    def _compute_letter_sequence_list(self, trip_list, default_letter_size):
        vsax_cuts = compute_cuts.compute_vsax_cuts(trip_list)
        letter_sequence_list = []
        for trip in trip_list:
            trip_letter_list = self._compute_letter_sequence_from_trip(
                trip, default_letter_size, vsax_cuts
            )
            letter_sequence_list.append(trip_letter_list)
        return letter_sequence_list

    @staticmethod
    def _compute_letter_sequence_from_trip(trip, default_letter_size, vsax_cuts):
        letter_list = []
        trip_len = trip.get_trip_size()
        for t in range(0, trip_len - default_letter_size + 1):
            segment_pointers = list(range(t, t + default_letter_size))
            segment_ts = trip.get_windown_obs(segment_pointers)
            current_letter = VSaxLetter(segment_ts, segment_pointers, vsax_cuts)
            if len(letter_list) == 0:
                letter_list.append(current_letter)
            else:
                previous_letter = letter_list[-1]
                if previous_letter.get_str_rep() == current_letter.get_str_rep():
                    concat_letter = current_letter.concat_with(previous_letter)
                    letter_list[-1] = concat_letter
                else:
                    letter_list.append(current_letter)
        return letter_list

    def _compute_word_sequence_from_trip(self, word_size, trip_index):
        letter_sequence = self.trip_letters(trip_index)
        word_list = []
        letter_sequence_len = len(letter_sequence)
        for t in range(0, letter_sequence_len - word_size + 1):
            letter_subseq = letter_sequence[t : t + word_size]
            vsax_word = VSaxWord(letter_subseq, trip_index)
            word_list.append(vsax_word)
        return word_list
