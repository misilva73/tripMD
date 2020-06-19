import numpy as np
from tripMD.dtwdist import compute_ndim_dtw_dist_mat
from tripMD.mdl import compute_mdl_cost


class Motif:
    """
    Class the motif object. A motif is a group of similar subsequences of a time-series. It contains a center, which is
    the representative subsequence of the motif and and set of member subsequences. In this implementation, each
    subsequence is a VSaxWord object with the same string representation and the motif can be also represented by that
    string pattern.

    Args:
        pattern (tuple of str): String representation of the motif. All vsax words that belong to the motif will
        same this pattern as their string representation.
        max_radius (float): maximum distance of all members to the center of the motif. If a vsax word with the same
        string representation as the motif's pattern has a DTW distance to the center higher than max_radius, then it
        won't be included as a member of the motif.
        compute_mdl (bool): boolean indicating whether the user wants to compute the MDL cost of the motif.

    Attributes:
        pattern (tuple of str): String representation of the motif. All vsax words that belong to the motif will
        same this pattern as their string representation.
        max_radius (float): maximum distance of all members to the center of the motif. If a vsax word with the same
        string representation as the motif's pattern has a distance to the center higher than max_radius, then it
        won't be included as a member of the motif.
        compute_mdl (bool): boolean indicating whether the user wants to compute the MDL cost of the motif.
        center (tripMD.vsax.objects.vsax_word.VSaxWord): vsax word at the center of the motif. It is the representative
        word of the motif.
        members (list of tripMD.vsax.objects.vsax_word.VSaxWord): List of all the vsax words that make up the motif.
        mean_dist (float): average of the DTW distances of all motif's members to the motif's center.
        mdl (float): Minimum description length cost of the motif (computed as defined by Tanaka et all).
        description (dict): User provided description for the motif. It defaults to None and must be defined through the
        function `add_description` .
    """

    def __init__(self, pattern, max_radius, compute_mdl):
        """
        Constructor of the Motif class

        Args:
            pattern (tuple of str): String representation of the motif. All vsax words that belong to the motif will
            same this pattern as their string representation.
            max_radius (float): maximum distance of all members to the center of the motif. If a vsax word with the same
            string representation as the motif's pattern has a distance to the center higher than max_radius, then it
            won't be included as a member of the motif.
            compute_mdl (bool): boolean indicating whether the user wants to compute the MDL cost of the motif.
        """
        self.pattern = pattern
        self.max_radius = max_radius
        self.compute_mdl = compute_mdl
        self.center = None
        self.members = None
        self.mean_dist = np.inf
        self.mdl = None
        self.description = None

    def compute_center_and_members(self, vsax_sequence, word_list):
        """
        Computation of the center and members of the motif that the provided pattern and max_radius. It does not return
        the final center and members. Instead it updates the corresponding attributes of the object (center, members,
        mean_dist and mdl, if the compute_mdl flag is set to True).

        Args:
            vsax_sequence (tripMD.vsax.objects.vsax_sequence.VSaxSequence): vsax sequence with the trips from which we
            are extracting the motif
            word_list (list of tripMD.vsax.objects.vsax_word.VSaxWord): Full list of words (which were extracted from
            the trips under analysis) from where we'll look for the motif's center and members.
        """
        candidate_word_list = self._get_candidates(word_list)
        dtw_dist_mat = self._compute_dtw_distance_matrix(
            vsax_sequence, candidate_word_list
        )
        members_count = 0
        for candidate_index in range(len(candidate_word_list)):
            candidate_center = candidate_word_list[candidate_index]
            candidate_members, candidate_mean_dist = self._compute_members_and_mean_dist(
                candidate_index, dtw_dist_mat, candidate_word_list
            )
            candidate_members_count = len(candidate_members)
            if (candidate_members_count > members_count) or (
                (candidate_members_count == members_count)
                and (candidate_mean_dist < self.mean_dist)
            ):
                members_count = candidate_members_count
                self.center = candidate_center
                self.members = candidate_members
                self.mean_dist = candidate_mean_dist
            else:
                continue
        if self.compute_mdl:
            self.mdl = compute_mdl_cost(self.members, word_list)

    def add_description(self, description):
        """
        Sets the motif's description
        Args:
            description (dict): User provided description for the motif
        """
        self.description = description

    def get_description(self):
        """
        Returns:
            description (dict): Dictionary with the user provided description for the motif
        """
        if self.description is None:
            raise Warning("Motif's description was not set")
        else:
            return self.description

    def get_members(self):
        """
        Returns:
            members (list of tripMD.vsax.objects.vsax_word.VSaxWord): List of all the vsax words that make up the motif.
        """
        return self.members

    def get_center(self):
        """
        Returns:
            center (tripMD.vsax.objects.vsax_word.VSaxWord): vsax word at the center of the motif. It is the representative
            word of the motif.
        """
        return self.center

    def get_pattern(self):
        """
        Returns:
            pattern (tuple of str): String representation of the motif. All vsax words that belong to the motif will
            same this pattern as their string representation.
        """
        return self.pattern

    def get_mean_distance(self):
        """
        Returns:
            mean_dist (float): average of the DTW distances of all motif's members to the motif's center.
        """
        return self.mean_dist

    def get_mdl_cost(self):
        """
        Returns:
            mdl (float): Minimum description length cost of the motif (computed as defined by Tanaka et all)
        """
        return self.mdl

    def print_summary(self):
        print("Pattern: " + str(self.pattern))
        print("Num of members: " + str(len(self.members)))
        print("Center size: " + str(len(self.center.get_pointers())))
        print("Mean distance: " + str(round(self.mean_dist, 4)))
        if self.compute_mdl:
            print("MDL cost: " + str(round(self.mdl, 2)))

    def get_members_obs_list(self, trip_list):
        """
        Args:
            trip_list (list of tripMD.objects.trip.Trip): list of original trips from which the motif was extracted

        Returns:
            member_obs_list (list):
        """
        member_obs_list = []
        if self.members is None:
            raise AttributeError("Motif's members are not yet computed")
        for member in self.members:
            member_pointers = member.get_pointers()
            member_trip_index = member.get_trip_index()
            member_trip = trip_list[member_trip_index]
            member_obs = member_trip.get_windown_obs(member_pointers)
            member_obs_list.append(member_obs)
        return member_obs_list

    def get_center_obs(self, trip_list):
        """

        Args:
            trip_list (list of tripMD.objects.trip.Trip): list of original trips from which the motif was extracted

        Returns:
            center_obs (list numpy.array):

        """
        if self.center is None:
            raise AttributeError("Motif's center is not yet computed")
        center_pointers = self.center.get_pointers()
        center_trip_index = self.center.get_trip_index()
        center_trip = trip_list[center_trip_index]
        center_obs = center_trip.get_windown_obs(center_pointers)
        return center_obs

    def get_members_labels_list(self, trip_list, label_index=0):
        """
        Args:
            trip_list (list of tripMD.objects.trip.Trip): list of original trips from which the motif was extracted
            label_index (int): index of the label one wishes to extract

        Returns:
            member_labels_list (list):
        """
        member_labels_list = []
        if self.members is None:
            raise AttributeError("Motif's members are not yet computed")
        for member in self.members:
            member_pointers = member.get_pointers()
            member_trip_index = member.get_trip_index()
            member_trip = trip_list[member_trip_index]
            member_labels = member_trip.get_window_labels(member_pointers, label_index)
            member_labels_list.append(member_labels)
        return member_labels_list

    def _get_candidates(self, word_list):
        """
        Args:
            word_list (list of tripMD.vsax.objects.vsax_word.VSaxWord): Full list of vsax words

        Returns:
            candidate_list (list of tripMD.vsax.objects.vsax_word.VSaxWord): List of vsax words with the same string
            representation as the motif's pattern

        """
        candidate_list = []
        for vsax_word in word_list:
            if vsax_word.get_str_rep() == self.pattern:
                candidate_list.append(vsax_word)
        return candidate_list

    def _compute_dtw_distance_matrix(self, vsax_sequence, word_list):
        """
        Args:
            vsax_sequence (tripMD.vsax.objects.vsax_sequence.VSaxSequence): vsax sequence with the trips from which we
            are extracting the motif
            word_list (list of tripMD.vsax.objects.vsax_word.VSaxWord): list of vsax words from which we wish to compute
            the pairwise DTW distances.

        Returns:
            dist_mat (numpy.array): matrix with the pairwise DTW distances of word_list.
        """
        ts_list = [vsax_sequence.get_word_obs(word) for word in word_list]
        dist_mat = compute_ndim_dtw_dist_mat(ts_list, self.max_radius)
        return dist_mat

    def _compute_members_and_mean_dist(self, center_index, dtw_dist_mat, word_list):
        """
        Args:
            center_index (int): index of the candidate to motif's center in in word_list.
            dtw_dist_mat (numpy.array): matrix with the pairwise DTW distances of word_list.
            word_list (list of tripMD.vsax.objects.vsax_word.VSaxWord): list of vsax words from which we computed the
            dtw_dist_mat and from where we will look for the motif's members, assuming the given center.

        Returns:
            final_members (list of tripMD.vsax.objects.vsax_word.VSaxWord): List with the member of the motif, assuming
            the given center.
            mean_dist (float): average of the DTW distances of all the members to the given center.
        """
        center_dist_vec = dtw_dist_mat[center_index]
        unpruned_members_index = [
            i for i, dist in enumerate(center_dist_vec) if dist < self.max_radius
        ]
        if len(unpruned_members_index) == 1:
            final_members = self.slice_list(word_list, unpruned_members_index)
            mean_dist = 0
        else:
            pruned_members_index = [unpruned_members_index[0]]
            for member_index in unpruned_members_index[1:]:
                member = word_list[member_index]
                last_pruned_member_index = pruned_members_index[-1]
                last_pruned_member = word_list[last_pruned_member_index]
                if self.lists_overlap(
                    last_pruned_member.get_pointers(), member.get_pointers()
                ):
                    last_pruned_member_dist = center_dist_vec[
                        last_pruned_member_index
                    ]
                    member_dist = center_dist_vec[member_index]
                    if member_dist < last_pruned_member_dist:
                        pruned_members_index = pruned_members_index[:-1]
                        pruned_members_index.append(member_index)
                    else:
                        continue
                else:
                    pruned_members_index.append(member_index)
            final_members = self.slice_list(word_list, pruned_members_index)
            mean_dist = np.mean(center_dist_vec[pruned_members_index])
        return final_members, mean_dist

    @staticmethod
    def lists_overlap(l1, l2):
        intersection_set = set(l1).intersection(set(l2))
        overlaping_test = len(intersection_set) > 0
        return overlaping_test

    @staticmethod
    def slice_list(original_list, index_list):
        sliced_list = []
        if len(index_list) == 0:
            return sliced_list
        else:
            for i in index_list:
                sliced_list.append(original_list[i])
        return sliced_list
