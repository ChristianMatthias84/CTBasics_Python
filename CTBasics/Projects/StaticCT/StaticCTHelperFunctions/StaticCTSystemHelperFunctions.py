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
            angles = StaticCTSystemHelperFunctions._Interleave(N_total, N_subsets, 360.0)
        elif ReconType == "180":
            if N_total % 2 != 0:
                raise ValueError("Fuer 180 muss N_total gerade sein")
            angles_half = StaticCTSystemHelperFunctions._Interleave(
                N_total // 2, N_subsets, 180.0
            )
            angles = angles_half + [a + 180.0 for a in angles_half]
        else:
            raise ValueError("ReconType unbekannt")

        angles = [a + global_offset_deg for a in angles]
        return np.deg2rad(np.array([a % 360.0 for a in angles]))

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
