from saxpy.paa import paa
from saxpy.sax import ts_to_string
import copy


class VSaxLetter(object):
    """
    Object for the letters of the variable sax representation. Each letter corresponds to a string discretization of a
    specific piece (or segment) of the a trips' multidimensional measurements.

    Args:
        segment_ts_list (list of numpy.array): list of numpy arrays with the original trip observations that make the
        letter. The length of the list corresponds to the size of the trip's segment while the length of the arrays
        correspond to the dimension of the observations (i.e., number of signals collected for the trip). Thus, each
        element of the list correspond to a single multidimensional observation of trip.
        segment_pointers (list of int): list with the indices of the letter in the original trip.
        cuts (dict of numpy.array): dictionary where the keys are the trips' dimensions and the values are that
        dimensions' vsax cuts

    Attributes:
        _ndim (int): number of dimensions of the letter, which correspond to the number of signals that make the
        multidimensional trip from wich the segment was taken.
        _pointers (list of int): list with the indices of the letter in the original trip.
        _cuts (dict of numpy.array): dictionary where the keys are the trips' dimensions and the values are that
        dimensions' vsax cuts
        _rep (tuple of str): string representation of the letter. For each signal in the multidimensional segment, a
        letter is computed according to the sax representation from the saxpy package and added to the tuple. Thus, each
        entry in the tuple corresponds to a single signal from the multidimensional trip's segment.
    """

    def __init__(self, segment_ts_list, segment_pointers, cuts):
        """
        Constructor for the class VSaxLetter.

        Args:
            segment_ts_list (list of numpy.array): list of numpy arrays with the original trip observations that make
            the letter. The length of the list corresponds to the size of the trip's segment while the length of the
            arrays correspond to the dimension of the observations (i.e., number of signals collected for the trip).
            Thus, each element of the list correspond to a single multidimensional observation of trip.
            segment_pointers (list of int): list with the indices of the letter in the original trip.
            cuts (dict of numpy.array): dictionary where the keys are the trips' dimensions and the values are that
            dimensions' vsax cuts
        """
        self._ndim = len(segment_ts_list[0])
        self._pointers = segment_pointers
        self._cuts = cuts
        self._rep = self._compute_str_rep(segment_ts_list)

    def _compute_str_rep(self, segment_ts_list):
        str_rep = []
        for dim in range(self._ndim):
            segment_ts_1d = self._get_1d_segment_ts(segment_ts_list, dim)
            str_rep_1d = self._compute_1_dim_str_rep(segment_ts_1d, dim)
            str_rep.append(str_rep_1d)
        return tuple(str_rep)

    @staticmethod
    def _get_1d_segment_ts(segment_ts_list, dim):
        segment_ts = []
        for obs_array in segment_ts_list:
            segment_ts.append(obs_array[dim])
        return segment_ts

    def _compute_1_dim_str_rep(self, segment_ts, dim):
        cuts = self._cuts[dim]
        paa_rep = paa(segment_ts, paa_segments=1)
        letter = ts_to_string(paa_rep, cuts)
        return letter

    def get_ndim(self):
        return self._ndim

    def get_str_rep(self, dim=None):
        if dim is None:
            return self._rep
        elif dim in list(range(self._ndim)):
            return self._rep[dim]
        else:
            raise IndexError("The provided dimension is out of range")

    def get_pointers(self):
        return self._pointers

    def set_pointers(self, new_pointers):
        self._pointers = new_pointers

    def concat_with(self, second_vsax_letter):
        if self._rep != second_vsax_letter.get_str_rep():
            raise Exception(
                "VSaxLetter concatenation only works if the two objects have the same str representation. "
                "Check this property by calling the `get_str_rep` method"
            )
        else:
            new_pointers = self._pointers + second_vsax_letter.get_pointers()
            concat_vsax_letter = copy.deepcopy(second_vsax_letter)
            concat_vsax_letter.set_pointers(new_pointers)
            return concat_vsax_letter
