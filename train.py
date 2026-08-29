"""Train a per-frame fall-anticipation model.

    python train.py --config configs/ours_preimpact.yaml
    python train.py --config configs/baseline_stgcn.yaml --seed 1
    python train.py --config configs/ours_preimpact.yaml --set train.epochs=5

Model selection happens on the **validation** split only, and it selects both
the weights and the operating point `(tau, k)`. Choosing a threshold on the test
set is the most common way an anticipation result gets quietly inflated -
sweeping thresholds and reporting the best is a test-set fit - so the threshold
is frozen here, written into the checkpoint, and `eval.py` applies it unchanged.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.build import build_datasets, load_clips
from data.datasets import collate_windows
from evaluation.decision import DecisionConfig, sweep_grid
from evaluation.metrics import evaluate, operating_curve, select_operating_point
from evaluation.runner import score_dataset
from losses.preimpact import build_loss
from models.build import build_model
from utils.config import apply_overrides, load_config, save_config
from utils.logging import (
    JsonlLogger,
    create_run_dir,
    environment_report,
    get_logger,
    save_json,
)
from utils.seed import resolve_device, seed_everything, worker_init_fn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="dotted config overrides, e.g. train.epochs=5")
    parser.add_argument("--seed", type=int, default=None, help="overrides run.seed")
    parser.add_argument("--fold", type=int, default=None, help="overrides data.split.fold")
    parser.add_argument("--run-name", default=None, help="overrides run.name")
    parser.add_argument("--out", default=None, help="overrides run.results_dir")
    parser.add_argument("--no-timestamp", action="store_true",
                        help="write to results/<name> instead of results/<name>_<time>")
    return parser.parse_args(argv)


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """Cosine decay with a linear warm-up, stepped per optimiser step."""
    epochs = int(cfg["epochs"])
    warmup_steps = int(cfg.get("warmup_epochs", 0)) * steps_per_epoch
    total_steps = max(epochs * steps_per_epoch, 1)

    if cfg.get("scheduler", "cosine") == "none":
        return None

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def validate(model, bundle, cfg, device) -> tuple[float, dict, DecisionConfig, bool]:
    """Score the validation clips and pick the best feasible operating point.

    Returns `(selection_score, report_row, decision, feasible)`.

    `feasible` is False when no `(tau, k)` on the grid met the false-alarm
    budget - common in the first few epochs. The selection score then falls back
    to `frame_auc - 1`, which is always below any feasible score (lead times are
    non-negative) yet still ranks the infeasible epochs against each other, so
    early stopping has something to work with instead of a flat -inf. The flag
    is returned rather than inferred, so the log can say which of the two
    quantities it is printing.
    """
    decision_cfg = cfg["decision"]
    w_pre = int(cfg["labels"]["w_pre"])
    early = decision_cfg.get("early_trigger", "false_alarm")
    default = DecisionConfig(
        threshold=float(decision_cfg["threshold"]),
        persistence=int(decision_cfg["persistence"]),
        refractory_frames=int(decision_cfg["refractory_frames"]),
    )

    scored = score_dataset(
        model, bundle.val_clips, device, amp=bool(cfg["train"].get("amp", False))
    )

    if not decision_cfg.get("select_on_val", True):
        row = evaluate(scored, default, w_pre, early).to_row()
        value = row.get(str(decision_cfg["objective"]), float("nan"))
        return (0.0 if not np.isfinite(value) else float(value)), row, default, True

    grid = sweep_grid(refractory_frames=int(decision_cfg["refractory_frames"]))
    rows = operating_curve(scored, grid, w_pre, early)
    best = select_operating_point(
        rows,
        max_false_alarms_per_hour=float(decision_cfg["max_false_alarms_per_hour"]),
        objective=str(decision_cfg["objective"]),
    )

    if best is None:
        fallback = evaluate(scored, default, w_pre, early)
        auc = fallback.frame_auc
        score = (0.0 if not np.isfinite(auc) else auc) - 1.0
        return score, fallback.to_row(), default, False

    chosen = DecisionConfig(
        threshold=float(best["decision"]["threshold"]),
        persistence=int(best["decision"]["persistence"]),
        refractory_frames=int(best["decision"]["refractory_frames"]),
    )
    value = best.get(str(decision_cfg["objective"]), float("nan"))
    score = 0.0 if not np.isfinite(value) else float(value)
    return score, best, chosen, True


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.set)

    if args.seed is not None:
        cfg["run"]["seed"] = args.seed
    if args.fold is not None:
        cfg["data"]["split"]["fold"] = args.fold
        cfg["data"]["split"]["mode"] = "folds"
    if args.run_name:
        cfg["run"]["name"] = args.run_name
    if args.out:
        cfg["run"]["results_dir"] = args.out

    seed = int(cfg["run"]["seed"])
    run_name = f"{cfg['run']['name']}_seed{seed}"
    run_dir = create_run_dir(cfg["run"]["results_dir"], run_name, not args.no_timestamp)
    logger = get_logger("train", run_dir)
    history = JsonlLogger(run_dir / "history.jsonl")

    seed_everything(seed, bool(cfg["run"].get("deterministic", True)))
    device = resolve_device(str(cfg["run"].get("device", "auto")))

    save_config(cfg, run_dir / "config.yaml")
    save_json(environment_report(), run_dir / "environment.json")
    logger.info("run directory: %s", run_dir)
    logger.info("device: %s | seed: %d", device, seed)

    # -- data ------------------------------------------------------------------
    clips = load_clips(cfg)
    bundle = build_datasets(cfg, clips)
    logger.info("%s", bundle.split.describe())
    logger.info(
        "clips: %d train / %d val / %d test | falls %d | ADL hours %.2f",
        bundle.stats["train"]["clips"],
        bundle.stats["val"]["clips"],
        bundle.stats["test"]["clips"],
        bundle.stats["all"]["falls"],
        bundle.stats["all"]["adl_hours"],
    )
    save_json(
        {"split": bundle.split.__dict__, "stats": bundle.stats}, run_dir / "split.json"
    )

    train_cfg = cfg["train"]
    loader = DataLoader(
        bundle.train,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_windows,
        drop_last=len(bundle.train) > int(train_cfg["batch_size"]),
        worker_init_fn=worker_init_fn,
        pin_memory=device.type == "cuda",
    )

    # -- model, loss, optimiser ------------------------------------------------
    model = build_model(cfg["model"], in_channels=bundle.features_cfg.in_channels).to(device)
    receptive_field = getattr(model.config, "receptive_field", None)
    logger.info(
        "model: %s | %d parameters | receptive field %s frames | window %d",
        cfg["model"].get("name", "stgcn"),
        model.num_parameters(),
        receptive_field if receptive_field is not None else "n/a",
        bundle.features_cfg.window,
    )
    if receptive_field is not None and receptive_field > bundle.features_cfg.window:
        # Not wrong - causal padding handles it - but the deepest blocks then
        # spend most of their kernel on padding rather than on data, which is
        # parameters and latency spent for nothing.
        logger.warning(
            "receptive field (%d) exceeds the window (%d): the deepest temporal "
            "kernels mostly see left-padding. Widen features.window or reduce "
            "model.blocks/kernel_size.",
            receptive_field, bundle.features_cfg.window,
        )

    criterion = build_loss(cfg["loss"])
    if criterion.config.pos_weight is None:
        prior = bundle.train.positive_fraction()
        weight = criterion.set_pos_weight_from_prior(prior)
        logger.info("positive frame fraction %.4f -> pos_weight %.2f", prior, weight)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = build_scheduler(optimizer, train_cfg, max(len(loader), 1))
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # -- training loop ---------------------------------------------------------
    best_score = -float("inf")
    best_epoch = -1
    best_decision = DecisionConfig(
        threshold=float(cfg["decision"]["threshold"]),
        persistence=int(cfg["decision"]["persistence"]),
        refractory_frames=int(cfg["decision"]["refractory_frames"]),
    )
    patience = int(train_cfg.get("patience", 12))
    eval_every = int(train_cfg.get("eval_every", 1))
    epochs = int(train_cfg["epochs"])

    for epoch in range(1, epochs + 1):
        model.train()
        started = time.time()
        running = 0.0
        batches = 0

        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            tti = batch["tti"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(features)
                loss, _ = criterion(logits.float(), labels, tti)

            scaler.scale(loss).backward()
            if float(train_cfg.get("grad_clip", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_cfg["grad_clip"])
                )
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            running += float(loss.item())
            batches += 1

        train_loss = running / max(batches, 1)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }

        if epoch % eval_every == 0 or epoch == epochs:
            score, row, decision, feasible = validate(model, bundle, cfg, device)
            record.update({"val_score": score, "val_feasible": feasible,
                           **{f"val_{k}": v for k, v in row.items()
                              if not isinstance(v, (dict, list))}})
            budget = cfg["decision"]["max_false_alarms_per_hour"]
            logger.info(
                "epoch %3d | loss %.4f | %s %.4f%s | recall %.3f | lead %.3fs | FA/h %.2f",
                epoch, train_loss,
                cfg["decision"]["objective"] if feasible else "fallback(auc-1)",
                score,
                "" if feasible else f" [no operating point under {budget} FA/h]",
                row["recall"], row["mean_lead_time"], row["false_alarms_per_hour"],
            )

            if score > best_score:
                best_score, best_epoch, best_decision = score, epoch, decision
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg,
                        "epoch": epoch,
                        "val_score": score,
                        "val_report": row,
                        "decision": {
                            "threshold": decision.threshold,
                            "persistence": decision.persistence,
                            "refractory_frames": decision.refractory_frames,
                        },
                        "split": bundle.split.__dict__,
                        "in_channels": bundle.features_cfg.in_channels,
                    },
                    run_dir / "best.pt",
                )
                logger.info(
                    "  new best (epoch %d): tau=%.2f k=%d",
                    epoch, decision.threshold, decision.persistence,
                )
        else:
            logger.info("epoch %3d | loss %.4f", epoch, train_loss)

        history.log(**record)

        if best_epoch > 0 and epoch - best_epoch >= patience:
            logger.info("early stop: %d epochs without improvement", patience)
            break

    if best_epoch < 0:
        logger.warning("no epoch produced a validation score; saving the last weights")
        torch.save(
            {"model": model.state_dict(), "config": cfg, "epoch": epochs,
             "in_channels": bundle.features_cfg.in_channels,
             "split": bundle.split.__dict__,
             "decision": {"threshold": best_decision.threshold,
                          "persistence": best_decision.persistence,
                          "refractory_frames": best_decision.refractory_frames}},
            run_dir / "best.pt",
        )

    save_json(
        {
            "best_epoch": best_epoch,
            "best_val_score": best_score,
            "objective": cfg["decision"]["objective"],
            "selected_decision": {
                "threshold": best_decision.threshold,
                "persistence": best_decision.persistence,
                "refractory_frames": best_decision.refractory_frames,
            },
        },
        run_dir / "training_summary.json",
    )
    logger.info(
        "done | best epoch %d | val %s %.4f | checkpoint %s",
        best_epoch, cfg["decision"]["objective"], best_score, run_dir / "best.pt",
    )
    return run_dir


if __name__ == "__main__":
    main()
