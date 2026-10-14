"""Shared classification steps and checkpoint state."""
import math
import random

import numpy as np
import torch
from torch import nn


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(state[0], state[1].tolist(), *state[2:]),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    name, values, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(values, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def learning_rate(step, total, warmup, peak, minimum):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


def train_epoch(model, batches, optimizer, args, epoch, steps_per_epoch):
    model.train()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer.zero_grad(set_to_none=True)
    total_loss, main_total, aux_total, count, updates = 0.0, 0.0, 0.0, 0, 0
    for i, (images, target) in enumerate(batches):
        images, target = images.to(args.device), target.to(args.device)
        start = (i // args.accum_steps) * args.accum_steps
        end = min(start + args.accum_steps, len(batches))
        # Account for both the final short accumulation group and a partial last batch.
        group_samples = min(end * batches.batch_size, len(batches.dataset)) - start * batches.batch_size
        if i == start:
            lr = learning_rate(epoch * steps_per_epoch + updates, args.epochs * steps_per_epoch,
                               args.warmup_epochs * steps_per_epoch, args.lr, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
        output = model(images)
        outputs = [output] if isinstance(output, torch.Tensor) else output
        main_loss = loss_fn(outputs[0], target)
        aux_loss = sum((loss_fn(pred, target) for pred in outputs[1:]), main_loss.new_zeros(()))
        loss = main_loss + args.aux_loss_weight * aux_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch {i}")
        (loss * images.shape[0] / group_samples).backward()
        if i + 1 == end:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        total_loss += loss.item() * images.shape[0]
        main_total += main_loss.item() * images.shape[0]
        aux_total += aux_loss.item() * images.shape[0]
        count += images.shape[0]
    return dict(train_loss=total_loss / count, train_main_loss=main_total / count,
                train_aux_loss=aux_total / count, lr=lr, optimizer_steps=updates)


@torch.inference_mode()
def evaluate(model, batches, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for images, target in batches:
        images, target = images.to(device), target.to(device)
        output = model(images)
        if not isinstance(output, torch.Tensor):
            output = output[0]
        total_loss += nn.functional.cross_entropy(output, target, reduction="sum").item()
        correct += (output.argmax(dim=1) == target).sum().item()
        count += images.shape[0]
    if count == 0:
        raise ValueError("Empty evaluation dataset")
    return dict(loss=total_loss / count, top1_acc=100.0 * correct / count, samples=count)


def save_checkpoint(path, model, optimizer, epoch, best, args, split_hash):
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                 best_acc=best, config=vars(args), split_hash=split_hash, rng=rng_state())
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def read_checkpoint(path, model, args, split_hash, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    ignored = {"resume", "eval_checkpoint", "output", "data_root", "download", "device", "workers",
               "latency_warmup", "latency_repeats", "threads"}
    for key, value in checkpoint["config"].items():
        if key not in ignored and getattr(args, key) != value:
            raise ValueError(f"Checkpoint config mismatch: {key}: {value!r} != {getattr(args, key)!r}")
    if checkpoint["split_hash"] != split_hash:
        raise ValueError("Checkpoint data split does not match")
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"])
    return checkpoint
