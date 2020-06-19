class VSaxWord(object):
    """
    Object for the words of the variable sax representation. In short, a variable sax word results from a union of variable
     sax letters.

    Args:
        vsax_letter_list (list of tripMD.vsax.objects.vsax_letter.VSaxLetter): list of variable sax letters that will
        make the word.
        trip_index (int): index to the trip from which the letters were built.

    Attributes:
        _trip_index (int): index to the trip from which the word was built (equal to the `trip_index` provided in the
        constructor).
        _ndim (int): number of dimensions of the word, which correspond to the number of signals that make the
        multidimensional trip from wich the segment was taken.
        _pointers (int): list with the indices of the word in the original trip.
        _rep (tuple of str): string representation of the word. For each signal in the multidimensional segment, a word
        is computed (as a concatenation of the respective 1-dim letters) and added to the tuple. Thus, each entry
        in the tuple corresponds to a single signal from the multidimensional trip's segment.
        _word_lengths (list of int): list with the length of the each letter in the word. In other words, it is a list
        with the number of pointers of each letter in the word.
    """

    def __init__(self, vsax_letter_list, trip_index):
        """
        Constructor for the VSaxWord class.

        Args:
            vsax_letter_list (list of tripMD.vsax.objects.vsax_letter.VSaxLetter): list of variable sax letters that will
            make the word.
            trip_index (int): index to the trip from which the letters were built.
        """
        self._trip_index = trip_index
        self._init_ndim(vsax_letter_list)
        self._init_pointers(vsax_letter_list)
        self._init_str_rep(vsax_letter_list)
        self._init_word_lengths(vsax_letter_list)

    def _init_ndim(self, vsax_letter_list):
        ndim_list = [vsax_letter.get_ndim() for vsax_letter in vsax_letter_list]
        if len(set(ndim_list)) != 1:
            raise Exception(
                "The dimension of the vsax letters is inconsistent. "
                "Make sure all vsax letters have the same number of features"
            )
        self._ndim = ndim_list[0]

    def _init_pointers(self, vsax_letter_list):
        pointers_list = []
        for vsax_letter in vsax_letter_list:
            letter_pointers = vsax_letter.get_pointers()
            pointers_list = pointers_list + letter_pointers
        self._pointers = sorted(set(pointers_list))

    def _init_str_rep(self, vsax_letter_list):
        word_list = []
        for dim in range(self._ndim):
            letter_rep_list = [
                vsax_letter.get_str_rep(dim) for vsax_letter in vsax_letter_list
            ]
            word = "".join(letter_rep_list)
            word_list.append(word)
        self._rep = tuple(word_list)

    def _init_word_lengths(self, vsax_letter_list):
        self._word_lengths = []
        for letter in vsax_letter_list:
            letter_length = len(letter.get_pointers())
            self._word_lengths.append(letter_length)

    def get_pointers(self):
        return self._pointers

    def get_trip_index(self):
        return self._trip_index

    def get_str_rep(self):
        return self._rep

    def get_word_lengths(self):
        return self._word_lengths
