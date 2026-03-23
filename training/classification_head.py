from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from trl.trainer.sft_trainer import SFTTrainer


class ClsHeadSFTTrainer(SFTTrainer):
    """
    SFTTrainer + extra CE loss on a classification token.

    The class label is represented by one of a small set of special tokens,
    e.g. <cls_up_by_1_2>.

    Important:
    Classification is computed over ONLY the class-token subset,
    not over the full vocabulary.
    """

    def __init__(
        self,
        *args,
        cls_lambda: float,
        cls_token_ids: List[int],
        **kwargs,
    ):
        kwargs.setdefault("compute_metrics", self._compute_cls_metrics)
        kwargs.setdefault(
            "preprocess_logits_for_metrics",
            self._preprocess_logits_for_metrics,
        )
        super().__init__(*args, **kwargs)

        self.cls_lambda = float(cls_lambda)

        # Stable ordering matters
        self.cls_token_ids_list = sorted(int(x) for x in cls_token_ids)
        self.cls_token_ids_set = set(self.cls_token_ids_list)

        # token_id -> class_index in [0, num_classes)
        self.token_id_to_class_idx: Dict[int, int] = {
            tok_id: i for i, tok_id in enumerate(self.cls_token_ids_list)
        }

    def _find_cls_positions(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Find first class-token position in each row.
        Returns -1 if missing.
        """
        batch_size, _ = ids.shape
        positions = torch.full(
            (batch_size,),
            fill_value=-1,
            dtype=torch.long,
            device=ids.device,
        )

        for b in range(batch_size):
            row = ids[b]
            hits = []
            for cls_id in self.cls_token_ids_list:
                idx = (row == cls_id).nonzero(as_tuple=False)
                if idx.numel() > 0:
                    hits.append(int(idx[0].item()))
            if hits:
                positions[b] = min(hits)

        return positions

    def _extract_cls_rows(
        self,
        logits: torch.Tensor,
        ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract rows for classification over class-token subset.

        Args:
            logits: [B, T, V]
            ids:    [B, T]

        Returns:
            cls_logits:       [N, C] logits restricted to class tokens
            target_class_idx: [N]    class indices in [0, C)
            valid_mask:       [B]
            pred_class_idx:   [N]
        """
        cls_positions = self._find_cls_positions(ids)
        valid_mask = cls_positions >= 1

        if not torch.any(valid_mask):
            empty_logits = torch.empty(
                (0, len(self.cls_token_ids_list)),
                device=logits.device,
                dtype=logits.dtype,
            )
            empty_targets = torch.empty(
                (0,),
                device=ids.device,
                dtype=torch.long,
            )
            empty_preds = torch.empty(
                (0,),
                device=ids.device,
                dtype=torch.long,
            )
            return empty_logits, empty_targets, valid_mask, empty_preds

        row_idx = torch.arange(ids.size(0), device=ids.device)[valid_mask]
        tok_idx = cls_positions[valid_mask]

        # Full-vocab logits for the position that predicts the cls token
        pred_logits_full = logits[row_idx, tok_idx - 1, :]   # [N, V]

        # Restrict to class-token subset only
        cls_token_id_tensor = torch.tensor(
            self.cls_token_ids_list,
            device=logits.device,
            dtype=torch.long,
        )
        cls_logits = pred_logits_full.index_select(dim=-1, index=cls_token_id_tensor)  # [N, C]

        # True token ids at the cls position
        target_token_ids = ids[row_idx, tok_idx]  # [N]

        # Map token ids -> class indices
        target_class_idx = torch.tensor(
            [self.token_id_to_class_idx[int(x.item())] for x in target_token_ids],
            device=ids.device,
            dtype=torch.long,
        )

        pred_class_idx = cls_logits.argmax(dim=-1)

        return cls_logits, target_class_idx, valid_mask, pred_class_idx

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Total loss:
            lm_loss + cls_lambda * cls_loss

        cls_loss is computed only over the class-token subset.
        """
        outputs = model(**inputs)
        lm_loss = outputs.loss

        input_ids = inputs["input_ids"]   # [B, T]
        logits = outputs.logits           # [B, T, V]

        cls_logits, target_class_idx, valid_mask, pred_class_idx = self._extract_cls_rows(
            logits=logits,
            ids=input_ids,
        )

        num_valid = int(valid_mask.sum().item())

        if num_valid == 0:
            cls_loss = torch.zeros((), device=lm_loss.device, dtype=lm_loss.dtype)
            cls_acc_value = 0.0
        else:
            cls_loss = F.cross_entropy(cls_logits, target_class_idx)
            cls_acc = (pred_class_idx == target_class_idx).float().mean()
            cls_acc_value = float(cls_acc.detach().cpu())

        total_loss = lm_loss + self.cls_lambda * cls_loss

        self.log(
            {
                "train/lm_loss": float(lm_loss.detach().cpu()),
                "train/cls_loss": float(cls_loss.detach().cpu()),
                "train/total_loss": float(total_loss.detach().cpu()),
                "train/cls_accuracy": cls_acc_value,
                "train/cls_rows_in_batch": num_valid,
                "train/batch_size": int(input_ids.size(0)),
            }
        )

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def _preprocess_logits_for_metrics(self, logits, labels):
        """
        Reduce eval payload to:
            pred_class_idx, target_class_idx, valid_mask
        """
        if isinstance(logits, tuple):
            logits = logits[0]

        cls_logits, target_class_idx, valid_mask, pred_class_idx = self._extract_cls_rows(
            logits=logits,
            ids=labels,
        )

        batch_size = labels.size(0)

        pred_out = torch.full(
            (batch_size,),
            fill_value=-100,
            device=labels.device,
            dtype=torch.long,
        )
        target_out = torch.full(
            (batch_size,),
            fill_value=-100,
            device=labels.device,
            dtype=torch.long,
        )

        if torch.any(valid_mask):
            row_idx = torch.arange(batch_size, device=labels.device)[valid_mask]
            pred_out[row_idx] = pred_class_idx
            target_out[row_idx] = target_class_idx

        packed = torch.stack(
            [pred_out, target_out, valid_mask.long()],
            dim=-1,
        )  # [B, 3]

        return packed

    def _compute_cls_metrics(self, eval_pred):
        """
        Eval classification metrics.
        """
        packed = eval_pred.predictions
        if isinstance(packed, tuple):
            packed = packed[0]

        packed = np.asarray(packed)

        pred_class_idx = packed[:, 0]
        target_class_idx = packed[:, 1]
        valid_mask = packed[:, 2].astype(bool)

        valid_count = int(valid_mask.sum())
        total_count = int(len(valid_mask))

        if valid_count == 0:
            return {
                "cls_accuracy": 0.0,
                "cls_valid_frac": 0.0,
                "cls_count": 0,
            }

        acc = float((pred_class_idx[valid_mask] == target_class_idx[valid_mask]).mean())

        return {
            "cls_accuracy": acc,
            "cls_valid_frac": valid_count / max(total_count, 1),
            "cls_count": valid_count,
        }

        