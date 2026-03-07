from __future__ import annotations

from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class ESMEmbedder:
    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D", device: str = "cpu") -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, sequences: List[str], batch_size: int = 16, max_len: int = 512) -> np.ndarray:
        all_vecs = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            toks = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            ).to(self.device)
            out = self.model(**toks)
            hidden = out.last_hidden_state
            mask = toks["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_vecs.append(pooled.cpu().numpy())
        return np.vstack(all_vecs)
