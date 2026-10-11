"""Subject-independent splitting (Section 14).

The proposal calls this the single choice that protects the validity of the
whole result, so the split is enforced structurally rather than left to
convention: every function here partitions **subjects**, and
`assert_subject_disjoint` is called by the dataset builder on every run. A clip
can only reach the test set if its subject is in the test subject list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .clips import ClipRecord


@dataclass(frozen=True)
class Split:
    """One train/val/test partition, stored as subject identifiers."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    fold: int = 0

    def subjects_for(self, stage: str) -> tuple[str, ...]:
        try:
            return {"train": self.train, "val": self.val, "test": self.test}[stage]
        except KeyError:
            raise ValueError(f"stage must be train/val/test, got {stage!r}") from None

    def describe(self) -> str:
        return (
            f"fold {self.fold}: "
            f"train={len(self.train)} val={len(self.val)} test={len(self.test)} subjects"
        )


def subject_folds(
    clips: list[ClipRecord],
    num_folds: int = 5,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> list[Split]:
    """Leave-subjects-out folds.

    Subjects are shuffled once with `seed` and dealt round-robin into folds, so
    the fold assignment is deterministic and independent of how many folds are
    requested from the same seed. The validation subjects are taken from the
    training pool of each fold, never from its test pool.

    Args:
        clips: every cached clip.
        num_folds: number of leave-subjects-out folds.
        val_fraction: fraction of the remaining subjects held out for early
            stopping and threshold selection.
        seed: shuffling seed.

    Returns:
        One `Split` per fold.
    """
    subjects = sorted({c.subject for c in clips})
    if len(subjects) < num_folds:
        raise ValueError(
            f"{len(subjects)} subjects cannot fill {num_folds} folds; "
            f"reduce num_folds or cache more data"
        )

    rng = np.random.default_rng(seed)
    order = list(rng.permutation(subjects))
    buckets: list[list[str]] = [order[i::num_folds] for i in range(num_folds)]

    splits: list[Split] = []
    for fold in range(num_folds):
        test = buckets[fold]
        pool = [s for i, b in enumerate(buckets) if i != fold for s in b]
        n_val = max(1, int(round(len(pool) * val_fraction)))
        # Rotate the pool per fold so the same subjects are not always validation.
        rotated = pool[fold % len(pool) :] + pool[: fold % len(pool)]
        val, train = rotated[:n_val], rotated[n_val:]
        if not train:
            raise ValueError("val_fraction leaves no training subjects")
        splits.append(
            Split(
                train=tuple(sorted(train)),
                val=tuple(sorted(val)),
                test=tuple(sorted(test)),
                fold=fold,
            )
        )
    return splits


def single_split(
    clips: list[ClipRecord],
    test_fraction: float = 0.2,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> Split:
    """One held-out split, for quick runs where full cross-validation is overkill."""
    subjects = sorted({c.subject for c in clips})
    if len(subjects) < 3:
        raise ValueError(f"need at least 3 subjects for a split, got {len(subjects)}")

    rng = np.random.default_rng(seed)
    order = list(rng.permutation(subjects))
    n_test = max(1, int(round(len(order) * test_fraction)))
    n_val = max(1, int(round(len(order) * val_fraction)))
    if n_test + n_val >= len(order):
        n_test, n_val = 1, 1

    test, val, train = order[:n_test], order[n_test : n_test + n_val], order[n_test + n_val :]
    return Split(
        train=tuple(sorted(train)), val=tuple(sorted(val)), test=tuple(sorted(test))
    )


def scene_family(view: str) -> str:
    """`coffee_room_01` -> `coffee_room`: the scene without its recording index."""
    return re.sub(r"_\d+$", "", view or "")


def matches_domain(clip: ClipRecord, domain: dict) -> bool:
    """True when the clip satisfies every key of `domain` (`source`, `scene`)."""
    unknown = set(domain) - {"source", "scene"}
    if unknown:
        raise ValueError(f"unknown domain keys {sorted(unknown)}; use 'source' and/or 'scene'")
    if not domain:
        raise ValueError("an empty domain would select every clip")
    if "source" in domain and clip.source != domain["source"]:
        return False
    if "scene" in domain and scene_family(clip.view) != domain["scene"]:
        return False
    return True


def holdout_split(
    clips: list[ClipRecord],
    test_domain: dict,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> Split:
    """Train on everything outside a domain, test on the domain.

    This is the realistic-data protocol. A random split inside one dataset lets
    the model see the same room, camera and (where actors are unlabelled) the
    same people on both sides, so it measures how well a model recognises a
    setting. Holding out a whole dataset (`{"source": "le2i"}`) or a whole scene
    (`{"source": "le2i", "scene": "coffee_room"}`) measures what a deployment
    faces: a place and a camera it has never seen.

    A subject with clips on both sides is an error and not something to quietly
    resolve, because dropping or moving it would change what the test measures.
    Validation subjects come from the training side only.
    """
    in_test = [matches_domain(c, test_domain) for c in clips]
    test_subjects = {c.subject for c, t in zip(clips, in_test) if t}
    train_pool = {c.subject for c, t in zip(clips, in_test) if not t}

    if not test_subjects:
        raise ValueError(f"no clips match test domain {test_domain}")
    shared = test_subjects & train_pool
    if shared:
        raise ValueError(
            f"subjects {sorted(shared)[:5]} have clips both inside and outside "
            f"{test_domain}; the held-out domain is not separable by subject"
        )
    if len(train_pool) < 3:
        raise ValueError(f"only {len(train_pool)} training subjects outside {test_domain}")

    rng = np.random.default_rng(seed)
    order = list(rng.permutation(sorted(train_pool)))
    n_val = max(1, int(round(len(order) * val_fraction)))
    if n_val >= len(order):
        raise ValueError("val_fraction leaves no training subjects")
    return Split(
        train=tuple(sorted(order[n_val:])),
        val=tuple(sorted(order[:n_val])),
        test=tuple(sorted(test_subjects)),
    )


def filter_clips(clips: list[ClipRecord], subjects) -> list[ClipRecord]:
    """Clips belonging to the given subjects."""
    wanted = set(subjects)
    return [c for c in clips if c.subject in wanted]


def assert_subject_disjoint(split: Split) -> None:
    """Raise if any subject appears in more than one partition.

    Called on every dataset construction. A silent overlap here would make every
    number in the paper measure identity memorisation instead of anticipation,
    and it is the kind of bug that produces suspiciously good results rather
    than an obvious crash.
    """
    pairs = (
        ("train", "val", set(split.train) & set(split.val)),
        ("train", "test", set(split.train) & set(split.test)),
        ("val", "test", set(split.val) & set(split.test)),
    )
    for a, b, overlap in pairs:
        if overlap:
            raise AssertionError(
                f"subject leakage between {a} and {b}: {sorted(overlap)}"
            )
