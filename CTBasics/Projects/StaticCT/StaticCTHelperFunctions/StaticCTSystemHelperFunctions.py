class StaticCTSystemHelperFunctions:
    """Angular sampling helpers for the static CT source ring."""

    @staticmethod
    def GreedyGapSamplingSubset(N, offset, width):
        """Picks N angles inside [offset, offset+width] by always halving the widest gap."""
        angles = [0.0] * max(int(N), 0)

        current = [offset, offset + width]
        first = offset + width / 2.0
        if angles:
            angles[0] = first % 360.0
        else:
            # MATLAB grows the preallocated empty array, so N=0 still yields one angle.
            angles.append(first % 360.0)
        current = sorted(current + [first])

        for k in range(2, N + 1):
            diffs = [current[i + 1] - current[i] for i in range(len(current) - 1)]
            idx = max(range(len(diffs)), key=lambda i: diffs[i])
            new_angle = current[idx] + diffs[idx] / 2.0
            angles[k - 1] = new_angle % 360.0
            current = sorted(current + [new_angle])

        return [a % 360.0 for a in angles]

    @staticmethod
    def SubsetInterleaving(N_total, N_subsets, ReconType, global_offset_deg=0):
        """Interleaves N_subsets greedy subsets so that every prefix stays well distributed."""
        import numpy as np

        ReconType = str(ReconType)

        if ReconType == "360":
            if N_total <= N_subsets:
                return StaticCTSystemHelperFunctions._LinearFallback(
                    N_total, global_offset_deg,
                    f"N_total ({N_total}) is not larger than N_subsets ({N_subsets}) in 360 "
                    "mode. Falling back to linear scan order.")
            angles = StaticCTSystemHelperFunctions._Interleave(N_total, N_subsets, 360.0)
        elif ReconType == "180":
            if N_total % 2 != 0:
                raise ValueError("Fuer 180 muss N_total gerade sein")
            if N_total // 2 < N_subsets:
                return StaticCTSystemHelperFunctions._LinearFallback(
                    N_total, global_offset_deg,
                    f"N_total/2 ({N_total // 2}) is smaller than N_subsets ({N_subsets}) in "
                    "180 mode. Falling back to linear scan order.")
            angles_half = StaticCTSystemHelperFunctions._Interleave(
                N_total // 2, N_subsets, 180.0
            )
            angles = angles_half + [a + 180.0 for a in angles_half]
        else:
            raise ValueError("ReconType unbekannt")

        angles = [a + global_offset_deg for a in angles]
        return np.deg2rad(np.array([a % 360.0 for a in angles]))

    @staticmethod
    def _LinearFallback(N_total, global_offset_deg, message):
        import warnings

        import numpy as np

        warnings.warn(message, stacklevel=3)
        angles = np.linspace(0, 360, N_total + 1)[:-1] + global_offset_deg
        return np.deg2rad(np.mod(angles, 360))

    @staticmethod
    def GreedyGapSamplingSubset_idx(N, offset, width):
        """Picks N distinct integers from [offset, offset+width-1] by splitting the largest gap.

        Integer fence posts (offset-1 and offset+width) bound the gaps, so no index repeats
        and N == width yields a full permutation of the window. Values are returned as given
        by `offset` (MATLAB callers pass 1-based indices).
        """
        if N > width:
            raise ValueError("N darf width (Anzahl verfuegbarer Indizes) nicht ueberschreiten")

        angles_idx = []
        fences = [offset - 1, offset + width]
        for _ in range(int(N)):
            gapSizes = [fences[i + 1] - fences[i] - 1 for i in range(len(fences) - 1)]
            gapIdx = max(range(len(gapSizes)), key=lambda i: gapSizes[i])
            lo = fences[gapIdx]
            hi = fences[gapIdx + 1]
            mid = lo + (hi - lo) // 2
            angles_idx.append(mid)
            fences = sorted(fences + [mid])
        return angles_idx

    @staticmethod
    def SubsetInterleaving_idx(N_total, N_subsets, ReconType, global_offset_deg=0,
                               angleArray=None):
        """Reorders angleArray by the index-domain counterpart of SubsetInterleaving.

        The subset windows tile the index range exactly (window size == subset count), so
        every index is visited once even if N_total does not divide by N_subsets.
        """
        import warnings

        import numpy as np

        angleArray = np.asarray(angleArray, dtype=float).ravel()
        ReconType = str(ReconType)

        if ReconType == "360":
            if N_total <= N_subsets:
                warnings.warn(
                    f"N_total ({N_total}) is not larger than N_subsets ({N_subsets}) in 360 "
                    "mode. Falling back to linear scan order.", stacklevel=2)
                return angleArray[:N_total] + global_offset_deg
            angles_idx = StaticCTSystemHelperFunctions._Interleave_idx(N_total, N_subsets)
        elif ReconType == "180":
            if N_total % 2 != 0:
                raise ValueError("Fuer 180 muss N_total gerade sein")
            N_half = N_total // 2
            if N_half < N_subsets:
                warnings.warn(
                    f"N_total/2 ({N_half}) is smaller than N_subsets ({N_subsets}) in 180 "
                    "mode. Falling back to linear scan order.", stacklevel=2)
                return angleArray[:N_total] + global_offset_deg
            angles_half_idx = StaticCTSystemHelperFunctions._Interleave_idx(N_half, N_subsets)
            angles_idx = angles_half_idx + [i + N_half for i in angles_half_idx]
        else:
            raise ValueError("ReconType unbekannt")

        # The indices are 1-based like MATLAB's.
        return angleArray[np.asarray(angles_idx) - 1] + global_offset_deg

    @staticmethod
    def _Interleave_idx(N_items, N_subsets):
        base = N_items // N_subsets
        remainder = N_items % N_subsets

        subsets = []
        cumOffset = 1
        for s in range(1, N_subsets + 1):
            N_s = base + 1 if s <= remainder else base
            subsets.append(
                StaticCTSystemHelperFunctions.GreedyGapSamplingSubset_idx(N_s, cumOffset, N_s)
            )
            cumOffset += N_s

        return StaticCTSystemHelperFunctions._RoundRobin(subsets)

    @staticmethod
    def _Interleave(N_items, N_subsets, span):
        width = span / N_subsets
        base = N_items // N_subsets
        remainder = N_items % N_subsets

        subsets = []
        for s in range(1, N_subsets + 1):
            N_s = base + 1 if s <= remainder else base
            offset = (s - 1) * width
            subsets.append(
                StaticCTSystemHelperFunctions.GreedyGapSamplingSubset(N_s, offset, width)
            )

        return StaticCTSystemHelperFunctions._RoundRobin(subsets)

    @staticmethod
    def _RoundRobin(subsets):
        angles = []
        k = 1
        finished = False
        while not finished:
            finished = True
            for subset in subsets:
                if len(subset) >= k:
                    angles.append(subset[k - 1])
                    finished = False
            k += 1

        return angles
