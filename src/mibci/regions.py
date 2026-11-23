"""Functional scalp regions of the 22-electrode BCI IV 2a montage.

The montage is a grid with 3.5 cm spacing over the sensorimotor strip:

                    Fz
         FC3  FC1  FCz  FC2  FC4
    C5   C3   C1   Cz   C2   C4   C6
         CP3  CP1  CPz  CP2  CP4
              P1   Pz   P2
                   POz

Motor imagery of one hand desynchronises mu/beta rhythms over the opposite hand area (C3
for the right hand, C4 for the left); feet imagery shows at the vertex (Cz), where the
foot area lies on the medial wall; tongue imagery involves the lateral, lower part of the
motor strip on both sides (towards C5/C6). Premotor and supplementary motor areas sit in
front (FC row, FCz/Fz), primary somatosensory cortex behind (CP row), and posterior
parietal cortex further back (P row, POz), where visual alpha also appears.

Two partitions are defined; both cover every electrode exactly once, and every coarse
region is a union of fine regions, so coarse coalitions are a subset of fine ones.

FINE (9):   each hemisphere split into premotor (FC), motor (C) and somatosensory (CP)
            rows; the midline split into front (Fz, FCz: supplementary motor area) and
            centre (Cz, CPz: foot area); parietal (P1, Pz, P2, POz).
COARSE (4): left sensorimotor, right sensorimotor, midline, parietal.
"""

from __future__ import annotations

from itertools import combinations

from mibci.data import EEG_CHANNELS

# Grid position (x to the right, y to the front) in units of the 3.5 cm electrode spacing.
POSITIONS = {
    "Fz": (0, 2),
    "FC3": (-2, 1), "FC1": (-1, 1), "FCz": (0, 1), "FC2": (1, 1), "FC4": (2, 1),
    "C5": (-3, 0), "C3": (-2, 0), "C1": (-1, 0), "Cz": (0, 0), "C2": (1, 0), "C4": (2, 0), "C6": (3, 0),
    "CP3": (-2, -1), "CP1": (-1, -1), "CPz": (0, -1), "CP2": (1, -1), "CP4": (2, -1),
    "P1": (-1, -2), "Pz": (0, -2), "P2": (1, -2),
    "POz": (0, -3),
}  # fmt: skip

FINE = {
    "L-premotor": ["FC3", "FC1"],
    "L-motor": ["C5", "C3", "C1"],
    "L-somatosensory": ["CP3", "CP1"],
    "R-premotor": ["FC2", "FC4"],
    "R-motor": ["C2", "C4", "C6"],
    "R-somatosensory": ["CP2", "CP4"],
    "mid-front": ["Fz", "FCz"],
    "mid-centre": ["Cz", "CPz"],
    "parietal": ["P1", "Pz", "P2", "POz"],
}
COARSE = {
    "left-sensorimotor": FINE["L-premotor"] + FINE["L-motor"] + FINE["L-somatosensory"],
    "right-sensorimotor": FINE["R-premotor"] + FINE["R-motor"] + FINE["R-somatosensory"],
    "midline": FINE["mid-front"] + FINE["mid-centre"],
    "parietal": FINE["parietal"],
}
PARTITIONS = {"coarse": COARSE, "fine": FINE}


def indices(names: list[str]) -> list[int]:
    """Montage indices of electrode names, in montage order (the order the network expects)."""
    return sorted(EEG_CHANNELS.index(n) for n in names)


def region_of(partition: dict[str, list[str]]) -> dict[str, str]:
    """Electrode name -> region name."""
    return {e: region for region, electrodes in partition.items() for e in electrodes}


def coalitions(partition: dict[str, list[str]]) -> list[tuple[str, ...]]:
    """Every non-empty set of regions, smallest first, regions in partition order."""
    names = list(partition)
    return [combo for size in range(1, len(names) + 1) for combo in combinations(names, size)]


def coalition_name(combo: tuple[str, ...]) -> str:
    return "+".join(combo)


def coalition_channels(partition: dict[str, list[str]], combo: tuple[str, ...]) -> list[int]:
    return indices([e for region in combo for e in partition[region]])


def neighbours(name: str) -> list[str]:
    """Electrodes one grid step away (up, down, left, right)."""
    x, y = POSITIONS[name]
    return [n for n, (u, v) in POSITIONS.items() if abs(u - x) + abs(v - y) == 1]


def _check() -> None:
    for label, partition in PARTITIONS.items():
        listed = [e for electrodes in partition.values() for e in electrodes]
        if sorted(listed) != sorted(EEG_CHANNELS) or len(listed) != len(set(listed)):
            raise AssertionError(f"{label} partition must cover each of the 22 electrodes once")
    if set(POSITIONS) != set(EEG_CHANNELS):
        raise AssertionError("every electrode needs a grid position")


_check()
