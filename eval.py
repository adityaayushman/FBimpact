"""Evaluate a trained checkpoint on its held-out test subjects.

    python eval.py --checkpoint results/ours_preimpact_seed0_*/best.pt
    python eval.py --checkpoint <run>/best.pt --explain --curve

The operating point comes from the checkpoint, where `train.py` froze it after
selecting it on validation. It is not re-tuned here. The threshold sweep printed
by `--curve` is reported as a curve precisely so that no single point on it has
to be defended as "the" threshold, and the point that *is* reported was chosen
without the test set.

The split also comes from the checkpoint rather than being recomputed, so a
later change to the splitting code cannot silently move subjects between train
and test for an already-trained model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.build import build_datasets, load_clips
from data.datasets import ClipDataset, FeatureConfig
from data.labels import LabelConfig
from data.splits import Split, assert_subject_disjoint, filter_clips
from evaluation.decision import DecisionConfig, sweep_grid
from evaluation.metrics import evaluate, operating_curve
from evaluation.runner import score_dataset
from models.build import build_model
from utils.config import apply_overrides
from utils.logging import format_table, get_logger, save_csv, save_json
from utils.seed import resolve_device, seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, help="path to best.pt")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="config overrides applied to the checkpoint's config")
    parser.add_argument("--explain", action="store_true",
                        help="run Stage E/F and report faithfulness")
    parser.add_argument("--curve", action="store_true",
                        help="also write the full operating-point sweep")
    parser.add_argument("--cache-dir", default=None,
                        help="evaluate on a different skeleton cache (transfer check)")
    parser.add_argument("--out", default=None, help="output directory (default: alongside the checkpoint)")
    return parser.parse_args(argv)


def load_checkpoint(path: str | Path, device: torch.device):
    """Load a checkpoint and rebuild its model."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = build_model(dict(cfg["model"]), in_channels=int(checkpoint.get("in_channels", 4)))
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    return model, cfg, checkpoint


def datasets_for(cfg: dict, checkpoint: dict, which: str, cache_dir: str | None):
    """Rebuild the requested split, preferring the checkpoint's own subject lists."""
    if cache_dir:
        cfg = dict(cfg)
        cfg["data"] = {**cfg.get("data", {}), "cache_dir": cache_dir}

    clips = load_clips(cfg)
    stored = checkpoint.get("split")

    if stored and not cache_dir:
        split = Split(
            train=tuple(stored["train"]),
            val=tuple(stored["val"]),
            test=tuple(stored["test"]),
            fold=int(stored.get("fold", 0)),
        )
        assert_subject_disjoint(split)
        subset = filter_clips(clips, split.subjects_for(which))
        if not subset:
            raise ValueError(
                f"no clips for the checkpoint's {which} subjects in this cache"
            )
        dataset = ClipDataset(
            subset,
            FeatureConfig(**dict(cfg["features"])),
            LabelConfig(**dict(cfg["labels"])),
        )
        return dataset, split

    # A different cache means different subjects: a cross-dataset transfer check
    # (Section 15, stretch goal). Every clip is unseen, so all of them are used.
    if cache_dir:
        dataset = ClipDataset(
            clips,
            FeatureConfig(**dict(cfg["features"])),
            LabelConfig(**dict(cfg["labels"])),
        )
        return dataset, Split(train=(), val=(), test=tuple(sorted({c.subject for c in clips})))

    bundle = build_datasets(cfg, clips)
    return {"test": bundle.test_clips, "val": bundle.val_clips}.get(
        which, bundle.test_clips
    ), bundle.split


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.out) if args.out else checkpoint_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("eval", out_dir)
    device = resolve_device("auto")
    model, cfg, checkpoint = load_checkpoint(checkpoint_path, device)
    cfg = apply_overrides(cfg, args.set)
    seed_everything(int(cfg["run"].get("seed", 0)), bool(cfg["run"].get("deterministic", True)))

    dataset, split = datasets_for(cfg, checkpoint, args.split, args.cache_dir)
    logger.info("checkpoint: %s (epoch %s)", checkpoint_path, checkpoint.get("epoch"))
    logger.info("evaluating %d clips from the %s split", len(dataset), args.split)

    stored_decision = checkpoint.get("decision") or cfg["decision"]
    decision = DecisionConfig(
        threshold=float(stored_decision["threshold"]),
        persistence=int(stored_decision["persistence"]),
        refractory_frames=int(stored_decision["refractory_frames"]),
    )
    logger.info(
        "operating point (frozen on validation): tau=%.2f k=%d refractory=%d",
        decision.threshold, decision.persistence, decision.refractory_frames,
    )

    w_pre = int(cfg["labels"]["w_pre"])
    early = cfg["decision"].get("early_trigger", "false_alarm")
    scored = score_dataset(model, dataset, device, amp=False)
    report = evaluate(scored, decision, w_pre, early,
                      int(cfg["decision"].get("tolerance_frames", 0)))

    logger.info("%s", report.summary())
    logger.info(
        "\n%s",
        format_table(
            [{
                "metric": k,
                "value": v,
            } for k, v in report.to_row().items() if not isinstance(v, dict)],
            ["metric", "value"],
        ),
    )

    results = {"report": report.to_row(), "split": args.split}
    save_json(results["report"], out_dir / f"report_{args.split}.json")
    save_csv([o.__dict__ for o in report.outcomes], out_dir / f"outcomes_{args.split}.csv")

    if args.curve:
        rows = operating_curve(
            scored,
            sweep_grid(refractory_frames=decision.refractory_frames),
            w_pre,
            early,
        )
        flat = [{k: v for k, v in r.items() if not isinstance(v, dict)} |
                {"threshold": r["decision"]["threshold"],
                 "persistence": r["decision"]["persistence"]}
                for r in rows]
        save_csv(flat, out_dir / f"operating_curve_{args.split}.csv")
        logger.info("operating curve: %d points -> operating_curve_%s.csv",
                    len(flat), args.split)

    if args.explain and cfg.get("explain", {}).get("enabled", True):
        from explain.report import explain_dataset

        explain_cfg = cfg["explain"]
        warnings, faithfulness = explain_dataset(
            model=model,
            dataset=dataset,
            scored=scored,
            decision=decision,
            w_pre=w_pre,
            device=device,
            method=str(explain_cfg.get("method", "attention")),
            max_warnings=explain_cfg.get("max_warnings"),
            baseline=str(explain_cfg.get("baseline", "zero")),
            num_random=int(explain_cfg.get("num_random", 5)),
            seed=int(cfg["run"].get("seed", 0)),
        )
        top_k = int(explain_cfg.get("top_k", 3))
        logger.info("explained %d warnings", len(warnings))
        for warning in warnings[:10]:
            logger.info("  %s", warning.message(top_k))

        if faithfulness is not None:
            logger.info("faithfulness | %s", faithfulness.summary())
            if faithfulness.deletion_gap <= 0 and faithfulness.insertion_gap <= 0:
                logger.warning(
                    "the ranking does not beat a random joint ordering: this is the "
                    "negative result Section 18 anticipates, and should be reported "
                    "as one rather than retuned away"
                )
            results["faithfulness"] = faithfulness.to_row()
            save_json(faithfulness.to_row(), out_dir / f"faithfulness_{args.split}.json")

        save_csv(
            [{
                "clip_id": w.clip_id,
                "trigger_frame": w.trigger_frame,
                "lead_time": w.lead_time,
                "score": w.score,
                "method": w.relevance.method,
                "phrase": w.relevance.phrase(top_k),
                "top_joints": "; ".join(f"{n}:{s:.3f}" for n, s in w.relevance.top_k(top_k)),
                "deletion_gap": w.curves.deletion_gap if w.curves else None,
                "insertion_gap": w.curves.insertion_gap if w.curves else None,
            } for w in warnings],
            out_dir / f"warnings_{args.split}.csv",
        )

    save_json(results, out_dir / f"eval_{args.split}.json")
    logger.info("results written to %s", out_dir)
    return results


if __name__ == "__main__":
    main()
