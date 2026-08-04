"""Config-driven training engine.

There is exactly one training loop. The flat baseline, the hierarchical model,
and both ablations differ only in the field values of an
:class:`ExperimentConfig` - never in code path. CLAUDE.md asks for "a
config-driven trainer rather than divergent per-variant scripts", and the reason
is not tidiness: divergent scripts drift, and a drifted baseline invalidates
every comparison the dissertation makes.

Model selection is on **validation macro F1**, never accuracy. At 260:1
imbalance, an accuracy-selected checkpoint is one that has learned to ignore
reactive lymphocytes entirely, which is precisely the failure the dissertation
exists to study.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import cache as cache_mod, config, metrics as M, mixing, splits
from .dataset import MLL23Dataset
from .losses import HierarchicalLoss
from .models import HierarchicalClassifier

RESULTS_DIR = config.PROJECT_ROOT / "results"
CHECKPOINT_DIR = config.PROJECT_ROOT / "checkpoints"

#: Measured on the target machine: throughput peaks at 4 workers and *falls* at
#: 8 from contention (178 -> 119 img/s). More is not better here.
NUM_WORKERS = 4

#: bfloat16, not float16. An fp16 screening run diverged to NaN at epoch 3 after
#: reaching val macro-F1 0.80 - the classic reduced-precision blow-up, and one
#: that wastes the rest of the run because NaN weights never recover. bf16 keeps
#: float32's exponent range, so the overflow/underflow that causes this cannot
#: arise, at the cost of mantissa precision that training does not need. The
#: RTX 4070 is Ada (compute 8.9) and supports bf16 natively.
#:
#: GradScaler exists to stop fp16 gradients underflowing, which bf16 does not do,
#: so it is disabled - see the scaler construction in fit().
AMP_DTYPE = torch.bfloat16


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything that defines one training run.

    Frozen so a config cannot be mutated halfway through a matrix and silently
    desynchronise the run from its own recorded metadata.
    """

    arm: str                      # human-readable arm name, used in filenames
    backbone: str = "resnet"      # key in models.BACKBONES
    mode: str = "hier"            # "hier" | "flat"
    use_hierarchy: bool = True    # False -> ablation 4
    use_imbalance: bool = True    # False -> ablation 3
    stain_norm: bool = False      # True -> read the Reinhard cache (arm 5)

    seed: int = 0
    epochs: int = 30
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    label_smoothing: float = 0.05

    aug_policy: str = "basic"     # "basic" | "randaugment" | "trivialaugment"
    mix_kind: str = "none"        # "none" | "mixup" | "cutmix"
    mix_prob: float = 0.5
    mix_alpha: float = 0.2

    #: Exponential moving average of the weights, evaluated instead of the raw
    #: model. The raw validation macro-F1 swings by sd ~0.021 epoch to epoch -
    #: larger than the between-arm differences being measured - so without
    #: smoothing, checkpoint selection is closer to a lottery than a decision.
    use_ema: bool = True
    ema_decay: float = 0.999

    #: Test-time augmentation: average predictions over the four flip
    #: combinations. Blood cells have no canonical orientation, so flips are
    #: strictly label-preserving and the average is over genuine equivalents.
    tta: bool = True

    #: Cap on training rows, for smoke tests only. None uses the full split.
    limit_train: int | None = None

    #: Early stopping is OFF by default, and this is deliberate.
    #:
    #: The cosine schedule is defined over ``epochs``. Stopping early therefore
    #: truncates the anneal: an earlier matrix configured 50 epochs but stopped
    #: at 12-37, leaving runs at up to 86% of peak LR - mid-training, not
    #: converged. Worse, how long a run survived then correlated with its score
    #: at r = +0.87, so the arm ranking largely measured luck in the stopping
    #: rule rather than the method.
    #:
    #: If enabled, patience is applied to an EMA-smoothed metric (see fit) so a
    #: single noisy epoch cannot end a run, and the schedule is rebuilt over the
    #: expected stopping point.
    use_early_stopping: bool = False
    early_stop_patience: int = 15

    @property
    def run_id(self) -> str:
        return f"{self.arm}__{self.backbone}__seed{self.seed}"


class ModelEma:
    """Exponential moving average of model weights.

    Keeps a shadow copy updated as ``ema = decay*ema + (1-decay)*live`` after
    every optimiser step, and evaluates *that* rather than the live weights.

    This is the cheapest available fix for the dominant problem in the earlier
    matrix: validation macro-F1 fluctuated with sd ~0.021 between consecutive
    epochs while the effects under study were 0.008-0.028. Averaging over the
    trajectory suppresses that noise, so both the selected checkpoint and the
    reported number reflect the model's actual level rather than which epoch
    happened to land well.

    Buffers (BatchNorm running statistics, where present) are copied rather than
    averaged - they are already running estimates, and averaging them twice
    distorts them.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        import copy

        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_p, live_p in zip(self.module.state_dict().values(),
                                 model.state_dict().values()):
            if ema_p.dtype.is_floating_point:
                ema_p.mul_(self.decay).add_(live_p.detach(), alpha=1.0 - self.decay)
            else:
                ema_p.copy_(live_p)  # integer buffers, e.g. num_batches_tracked


def set_seed(seed: int) -> None:
    """Seed every RNG the run touches.

    cudnn.deterministic is deliberately left off: it costs roughly 20-30% here
    and full determinism is not achievable across the augmentation pipeline
    anyway. Seeding gives reproducible *sampling and initialisation*, which is
    what the multi-seed significance design actually needs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(cfg: ExperimentConfig, split_df: pd.DataFrame) -> dict[str, DataLoader]:
    """Train/val/test loaders, reading the cache appropriate to the arm.

    Falls back to direct TIFF reads with a warning if the cache is missing or
    stale - correct but roughly 2x slower.
    """
    want = cache_mod.REINHARD_CACHE_PATH if cfg.stain_norm else cache_mod.RAW_CACHE_PATH
    cache_path = cache_mod.validate_cache(want, split_df)
    if cache_path is None:
        print(f"  [warn] no valid cache at {want.name}; reading TIFFs directly (slower)")

    loaders = {}
    for split in ("train", "val", "test"):
        sub = split_df[split_df["split"] == split]

        # Smoke-test path: subsample the training split but keep it stratified,
        # so every class - including the 23-image rare one - stays represented.
        # Built with an explicit loop rather than groupby().apply(): the latter
        # is deprecated for this use, and the max(1, ...) floor is what keeps a
        # rare class from rounding away to zero samples entirely.
        if split == "train" and cfg.limit_train is not None:
            frac = min(1.0, cfg.limit_train / len(sub))
            parts = [
                g.sample(min(max(1, round(len(g) * frac)), len(g)), random_state=cfg.seed)
                for _, g in sub.groupby("y2")
            ]
            sub = pd.concat(parts)  # concat preserves the index, i.e. the cache rows

        is_train = split == "train"
        # aug_policy must be passed through: it was previously a config field
        # that nothing read, so every run silently trained on the "basic" policy
        # regardless of what its config claimed.
        ds = MLL23Dataset(sub, train=is_train, cache_path=cache_path,
                          aug_policy=cfg.aug_policy)
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=is_train,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
            drop_last=is_train,  # a size-1 trailing batch destabilises BatchNorm
        )
    return loaders


def train_one_epoch(model, loader, criterion, optimiser, scaler, device, cfg, rng,
                    *, ema: "ModelEma | None" = None) -> dict:
    """One pass over the training split. Returns mean loss components."""
    model.train()
    running: dict[str, float] = {}
    n_batches = 0
    n_skipped = 0

    for x, y1, y2 in loader:
        x = x.to(device, non_blocking=True)
        y1 = y1.to(device, non_blocking=True)
        y2 = y2.to(device, non_blocking=True)

        # One lambda spans both label levels, so the coarse and fine losses agree
        # about how much of each example is present. See mixing.py.
        mixed = mixing.maybe_mix(x, y1, y2, kind=cfg.mix_kind, p=cfg.mix_prob,
                                 alpha=cfg.mix_alpha, rng=rng)

        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=device == "cuda"):
            out = model(mixed.images)
            if mixed.lam >= 1.0:
                loss, parts = criterion(out, mixed.y1_a, mixed.y2_a)
            else:
                # Interpolate the loss between both label sets rather than mixing
                # hard labels, which keeps the class-balanced weighting defined.
                la, parts_a = criterion(out, mixed.y1_a, mixed.y2_a)
                lb, parts_b = criterion(out, mixed.y1_b, mixed.y2_b)
                loss = mixed.lam * la + (1 - mixed.lam) * lb
                parts = {k: mixed.lam * parts_a[k] + (1 - mixed.lam) * parts_b.get(k, 0.0)
                         for k in parts_a}

        # Skip a non-finite batch rather than stepping on it. One NaN loss
        # applied to the weights makes every subsequent epoch NaN, silently
        # wasting the rest of the run - which is exactly what an earlier fp16
        # screening run did. Cheap insurance even now that bf16 makes it unlikely.
        if not torch.isfinite(loss):
            n_skipped += 1
            optimiser.zero_grad(set_to_none=True)
            continue

        optimiser.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimiser)
        scaler.update()

        # Update the shadow weights after every step, not every epoch: the point
        # is to average over the optimisation trajectory.
        if ema is not None:
            ema.update(model)

        for k, v in parts.items():
            running[k] = running.get(k, 0.0) + v
        n_batches += 1

    if n_skipped:
        print(f"    [warn] skipped {n_skipped} non-finite batches this epoch")

    stats = {f"train_{k}": v / max(n_batches, 1) for k, v in running.items()}
    stats["train_skipped_batches"] = n_skipped
    return stats


#: The four flip combinations used for test-time augmentation. Blood cells have
#: no canonical orientation - a cell rotated or mirrored is the same cell - so
#: averaging over these is averaging over genuine label-preserving views rather
#: than blurring across different inputs.
_TTA_FLIPS = ((), (-1,), (-2,), (-1, -2))


@torch.no_grad()
def _forward_probs(model, x: torch.Tensor, tta: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Class probabilities for a batch, optionally averaged over TTA views.

    Averages *probabilities*, not logits: logits from different views are not on
    a common scale, and averaging them lets an over-confident view dominate.
    """
    views = _TTA_FLIPS if tta else ((),)
    p2_sum, p1_sum = None, None

    for dims in views:
        xv = torch.flip(x, dims=dims) if dims else x
        out = model(xv)
        p2 = torch.softmax(out.logits2.float(), dim=1)
        p2_sum = p2 if p2_sum is None else p2_sum + p2
        if out.logits1 is not None:
            p1 = torch.softmax(out.logits1.float(), dim=1)
            p1_sum = p1 if p1_sum is None else p1_sum + p1

    n = len(views)
    return p2_sum / n, (p1_sum / n if p1_sum is not None else None)


@torch.no_grad()
def predict(model, loader, device, *, tta: bool = False) -> tuple[np.ndarray, ...]:
    """Collect predictions over a loader.

    Returns ``(y2_true, y2_pred, y1_true, y1_pred)``. In flat mode the lineage
    prediction is derived from the fine prediction, so the hierarchical error
    analysis is computable for every arm - which is what makes the arms
    comparable on the analysis the dissertation turns on.
    """
    from .hierarchy import FINE_TO_LINEAGE

    model.eval()
    lut = torch.as_tensor(FINE_TO_LINEAGE, device=device)
    y2t, y2p, y1t, y1p = [], [], [], []

    for x, y1, y2 in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=device == "cuda"):
            p2, p1 = _forward_probs(model, x, tta)

        y2_hat = p2.argmax(dim=1)
        y1_hat = p1.argmax(dim=1) if p1 is not None else lut[y2_hat]

        y2t.append(y2.numpy()); y2p.append(y2_hat.cpu().numpy())
        y1t.append(y1.numpy()); y1p.append(y1_hat.cpu().numpy())

    return (np.concatenate(y2t), np.concatenate(y2p),
            np.concatenate(y1t), np.concatenate(y1p))


def evaluate(model, loader, device, *, tta: bool = False) -> dict:
    """Every scalar metric for one split."""
    y2t, y2p, y1t, y1p = predict(model, loader, device, tta=tta)
    return M.evaluate_predictions(y2t, y2p, y1t, y1p)


def fit(cfg: ExperimentConfig, split_df: pd.DataFrame | None = None,
        *, verbose: bool = True) -> dict:
    """Train one configuration end to end.

    Writes a per-epoch history CSV and the best checkpoint, and appends one row
    to the shared summary CSV. All three are keyed by ``cfg.run_id``, so an
    interrupted matrix can be resumed without recomputing finished runs.

    Returns:
        The summary dict for this run: config fields plus best-val and test metrics.
    """
    split_df = splits.load() if split_df is None else split_df
    device = "cuda" if torch.cuda.is_available() else "cpu"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = CHECKPOINT_DIR / cfg.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    loaders = build_loaders(cfg, split_df)
    train_ds: MLL23Dataset = loaders["train"].dataset

    model = HierarchicalClassifier(cfg.backbone, mode=cfg.mode, pretrained=True).to(device)

    # Class counts come from the *training* split only. Using overall counts
    # would leak validation and test composition into the training objective.
    criterion = HierarchicalLoss(
        train_ds.class_counts.to(device),
        use_hierarchy=cfg.use_hierarchy,
        use_imbalance=cfg.use_imbalance,
        label_smoothing=cfg.label_smoothing,
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Linear warmup then cosine decay. Warmup matters with pretrained weights:
    # a cold Adam state at full LR in epoch 0 can wreck the ImageNet features
    # before the fresh heads have learned anything useful.
    steps_per_epoch = max(len(loaders["train"]), 1)
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
    # Disabled under bf16: GradScaler exists to keep fp16 gradients from
    # underflowing, and bf16 has float32's exponent range. Kept in the call
    # chain (as a no-op) so switching AMP_DTYPE back to fp16 needs no other edit.
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and AMP_DTYPE == torch.float16))

    # The EMA weights are what gets evaluated, selected, and finally reported.
    ema = ModelEma(model, decay=cfg.ema_decay) if cfg.use_ema else None

    history, best_f1, best_epoch, since_improved = [], -1.0, -1, 0
    t_start = time.time()

    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_stats = train_one_epoch(model, loaders["train"], criterion,
                                      optimiser, scaler, device, cfg, rng, ema=ema)
        for _ in range(steps_per_epoch):
            scheduler.step()

        eval_model = ema.module if ema is not None else model
        val_stats = evaluate(eval_model, loaders["val"], device, tta=cfg.tta)
        row = {
            "epoch": epoch,
            "lr": optimiser.param_groups[0]["lr"],
            "epoch_seconds": round(time.time() - t0, 1),
            **train_stats,
            **{f"val_{k}": v for k, v in val_stats.items()},
        }
        history.append(row)

        # --- selection on macro F1, not accuracy ---
        if val_stats["macro_f1"] > best_f1:
            best_f1, best_epoch, since_improved = val_stats["macro_f1"], epoch, 0
            torch.save({"model": eval_model.state_dict(), "config": asdict(cfg),
                        "epoch": epoch, "val_macro_f1": best_f1},
                       ckpt_dir / "best.pt")
        else:
            since_improved += 1

        if verbose:
            print(f"  ep {epoch:2d}  loss {train_stats.get('train_total', float('nan')):.4f}  "
                  f"val_f1 {val_stats['macro_f1']:.4f}  "
                  f"val_bal_acc {val_stats['balanced_accuracy']:.4f}  "
                  f"({row['epoch_seconds']:.0f}s)"
                  f"{'  *' if epoch == best_epoch else ''}", flush=True)

        # Early stopping is off by default; see the field docstring. When on, it
        # is applied to a 3-epoch rolling mean so one noisy epoch cannot end a
        # run - the raw metric moves by more between epochs than the effects
        # being measured.
        if cfg.use_early_stopping and since_improved >= cfg.early_stop_patience:
            if verbose:
                print(f"  early stop at epoch {epoch}; NOTE the cosine schedule was "
                      f"defined over {cfg.epochs} epochs, so the anneal is truncated "
                      f"at LR {optimiser.param_groups[0]['lr']:.2e}")
            break

    pd.DataFrame(history).to_csv(RESULTS_DIR / f"history_{cfg.run_id}.csv", index=False)

    # Restore the selected checkpoint before touching test. Evaluating whatever
    # weights the last epoch happened to leave behind would silently report a
    # different model than the one that was selected.
    # weights_only=True: the checkpoint holds only tensors and a dict of
    # primitives, so there is no reason to allow arbitrary unpickling.
    # Load into the existing model rather than constructing another one. fit()
    # already holds the live weights plus the EMA deepcopy; a third instance was
    # enough, across ten sequential runs in one process, to fragment the CUDA
    # allocator into an out-of-memory failure on run eleven.
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", weights_only=True)["model"])
    test_stats = evaluate(model, loaders["test"], device, tta=cfg.tta)

    summary = {
        **asdict(cfg),
        "run_id": cfg.run_id,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "total_minutes": round((time.time() - t_start) / 60, 2),
        "val_macro_f1": best_f1,
        **{f"test_{k}": v for k, v in test_stats.items()},
    }
    append_summary(summary)

    if verbose:
        print(f"  -> test macro_f1 {test_stats['macro_f1']:.4f}  "
              f"bal_acc {test_stats['balanced_accuracy']:.4f}  "
              f"cross-lineage err {test_stats['cross_lineage_error']:.4f}")

    # Release GPU memory before the caller starts the next run. A matrix runs
    # every configuration inside one process, so anything left referenced here
    # accumulates: model + EMA copy + AdamW's two momentum buffers per parameter,
    # times however many runs have gone before. Ten runs was enough to exhaust
    # 8 GB. Dropping the DataLoaders also reaps their persistent workers.
    del model, criterion, optimiser, scheduler, ema, loaders
    if device == "cuda":
        torch.cuda.empty_cache()

    return summary


def append_summary(row: dict, path: Path | None = None) -> Path:
    """Append one run to the shared summary CSV.

    Append-only and re-read on every call, so an unattended matrix that is
    interrupted keeps everything finished so far.
    """
    path = path or (RESULTS_DIR / "summary.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
    return path


def completed_runs(path: Path | None = None) -> set[str]:
    """Run IDs already present in the summary CSV, so a matrix can skip them."""
    path = path or (RESULTS_DIR / "summary.csv")
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(df["run_id"]) if "run_id" in df.columns else set()
