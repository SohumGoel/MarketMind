from __future__ import annotations

from collections import Counter
from typing import Iterable

import torch


def debug_cls_dataset(
    dataset,
    tokenizer,
    cls_token_ids: Iterable[int],
    max_length: int,
    n_show: int = 5,
) -> None:
    """
    Debug whether class tokens survive tokenization/truncation.
    Expects dataset examples to contain a 'text' field.
    """
    cls_token_ids = set(int(x) for x in cls_token_ids)

    total = len(dataset)
    missing = 0
    truncated = 0
    lengths = []
    cls_positions = []
    shown = 0

    print("\n[debug_cls_dataset] ---- start ----")
    print(f"[debug_cls_dataset] dataset size: {total}")
    print(f"[debug_cls_dataset] max_length: {max_length}")
    print(f"[debug_cls_dataset] num cls tokens: {len(cls_token_ids)}")

    for i, ex in enumerate(dataset):
        text = ex["text"]

        full_ids = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        trunc_ids = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )["input_ids"]

        lengths.append(len(full_ids))
        if len(full_ids) > max_length:
            truncated += 1

        hits = [j for j, tok in enumerate(trunc_ids) if tok in cls_token_ids]
        if not hits:
            missing += 1
            if shown < n_show:
                print(f"\n[debug_cls_dataset] MISSING cls token after truncation at example {i}")
                print(f"label: {ex.get('label')}")
                print(f"label_token: {ex.get('label_token')}")
                print(f"full_len={len(full_ids)}, trunc_len={len(trunc_ids)}")
                print("TEXT HEAD:")
                print(text[:1200])
                print("TEXT TAIL:")
                print(text[-1200:])
                shown += 1
        else:
            cls_positions.append(hits[0])

    print("\n[debug_cls_dataset] summary")
    print(f"  truncated examples: {truncated}/{total} = {truncated / max(total,1):.3f}")
    print(f"  missing cls token:   {missing}/{total} = {missing / max(total,1):.3f}")

    if lengths:
        lens_sorted = sorted(lengths)
        for q in [50, 90, 95, 99]:
            idx = min(len(lens_sorted) - 1, int(len(lens_sorted) * q / 100))
            print(f"  p{q} full token length: {lens_sorted[idx]}")

    if cls_positions:
        pos_sorted = sorted(cls_positions)
        for q in [50, 90, 95, 99]:
            idx = min(len(pos_sorted) - 1, int(len(pos_sorted) * q / 100))
            print(f"  p{q} cls token position after truncation: {pos_sorted[idx]}")

    print("[debug_cls_dataset] ---- end ----\n")


def debug_cls_tokens(tokenizer, label_tokens, n_show: int = 20) -> None:
    """
    Verify each label token is really a single token.
    """
    print("\n[debug_cls_tokens] ---- start ----")
    bad = 0
    for tok in list(label_tokens)[:n_show]:
        ids = tokenizer(tok, add_special_tokens=False)["input_ids"]
        print(f"{tok:30s} -> {ids}")
        if len(ids) != 1:
            bad += 1

    if len(label_tokens) > n_show:
        print(f"... showing first {n_show} / {len(label_tokens)}")

    for tok in label_tokens:
        ids = tokenizer(tok, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            bad += 1

    print(f"[debug_cls_tokens] multi-token labels found: {bad}")
    print("[debug_cls_tokens] ---- end ----\n")


def debug_one_training_batch(
    trainer,
    tokenizer,
    n_batches: int = 2,
    topk: int = 10,
) -> None:
    """
    Inspect actual batches after collation, and show:
    - whether cls token exists
    - target cls token
    - top-k predictions at cls position
    """
    print("\n[debug_one_training_batch] ---- start ----")

    dl = trainer.get_train_dataloader()
    for batch_idx, batch in enumerate(dl):
        if batch_idx >= n_batches:
            break

        batch = trainer._prepare_inputs(batch)

        with torch.no_grad():
            outputs = trainer.model(**batch)

        input_ids = batch["input_ids"]
        logits = outputs.logits

        cls_positions = trainer._find_cls_positions(input_ids)
        valid_rows = cls_positions >= 1

        print(f"\n[batch {batch_idx}] batch_size={input_ids.size(0)}")
        print(f"[batch {batch_idx}] valid_rows={int(valid_rows.sum().item())}/{input_ids.size(0)}")

        for b in range(min(4, input_ids.size(0))):
            pos = int(cls_positions[b].item())
            print(f"\n  row={b} cls_pos={pos}")

            if pos < 1:
                print("  -> no cls token in this row")
                decoded = tokenizer.decode(input_ids[b], skip_special_tokens=False)
                print(decoded[:1000])
                continue

            target_id = int(input_ids[b, pos].item())
            pred_logits = logits[b, pos - 1]
            pred_id = int(pred_logits.argmax(dim=-1).item())

            cls_scores = []
            for cls_id in sorted(trainer.cls_token_ids_list):
                cls_scores.append((cls_id, float(pred_logits[cls_id].item())))

            cls_scores = sorted(cls_scores, key=lambda x: x[1], reverse=True)

            print("  cls-token scores:")
            for cls_id, score in cls_scores[:12]:
                tok = tokenizer.decode([cls_id], skip_special_tokens=False)
                star = " <== target" if cls_id == target_id else ""
                print(f"    id={cls_id} logit={score:9.4f} tok={tok!r}{star}")

            topv, topi = torch.topk(pred_logits, k=min(topk, pred_logits.numel()))
            topi = topi.tolist()
            topv = topv.tolist()

            print(f"  target_id={target_id} target_tok={tokenizer.decode([target_id], skip_special_tokens=False)!r}")
            print(f"  pred_id={pred_id} pred_tok={tokenizer.decode([pred_id], skip_special_tokens=False)!r}")
            print("  topk:")
            for rank, (tid, score) in enumerate(zip(topi, topv), start=1):
                tok = tokenizer.decode([tid], skip_special_tokens=False)
                star = " <== target" if tid == target_id else ""
                print(f"    {rank:2d}. id={tid:6d} logit={score:9.4f} tok={tok!r}{star}")

            decoded = tokenizer.decode(input_ids[b], skip_special_tokens=False)
            start = max(0, pos - 40)
            end = min(input_ids.size(1), pos + 40)
            window = input_ids[b, start:end].tolist()
            print("  local token window:")
            print(tokenizer.decode(window, skip_special_tokens=False))

            cls_ids = torch.tensor(trainer.cls_token_ids_list, device=pred_logits.device)
            cls_logits = pred_logits.index_select(dim=-1, index=cls_ids)
            cls_pred_class = int(cls_logits.argmax().item())
            cls_pred_token_id = trainer.cls_token_ids_list[cls_pred_class]

            print(f"  restricted_cls_pred_id={cls_pred_token_id} "
                f"restricted_cls_pred_tok={tokenizer.decode([cls_pred_token_id], skip_special_tokens=False)!r}")

    print("[debug_one_training_batch] ---- end ----\n")


def debug_eval_schedule(cfg) -> None:
    print("\n[debug_eval_schedule] ---- start ----")
    print(f"eval_strategy = {cfg.eval_strategy}")
    print(f"save_strategy = {cfg.save_strategy}")
    #print(f"logging_steps = {cfg.logging_steps}")
    print(f"max_steps = {getattr(cfg, 'max_steps', None)}")
    if hasattr(cfg, "eval_steps"):
        print(f"eval_steps = {cfg.eval_steps}")
    print("[debug_eval_schedule] ---- end ----\n")

def print_trainable_params(model):
    trainable = 0
    total = 0
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            print("[trainable]", n, tuple(p.shape))
    print(f"trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")


def debug_cls_row_grads(model, cls_token_ids):
    emb = model.get_input_embeddings().weight
    head = model.lm_head.weight

    print("\n[DEBUG cls row grad norms]")
    if emb.grad is None:
        print("embed grad is None")
    else:
        vals = [float(emb.grad[i].norm().detach().cpu()) for i in cls_token_ids]
        print("embed grad norms:", vals[:12])

    if head.grad is None:
        print("lm_head grad is None")
    else:
        vals = [float(head.grad[i].norm().detach().cpu()) for i in cls_token_ids]
        print("lm_head grad norms:", vals[:12])