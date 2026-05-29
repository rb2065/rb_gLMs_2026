import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import csv
import random
from datetime import datetime, date
import argparse
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pysam
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DNABERT_LOCAL_PATH = "/home/rb2065/rds/rds-berners-lee-RCkqvqaENjU/huggingface_boemo/models--MBoemo--DNABERT-2-117M-Flash/snapshots/8be1414aab8283f09008c17651bc60beae65d77b"
# adding to the example comment
class DNABERTChunkAttentionRegressor(nn.Module):
    def __init__(
        self,
        model_name=DNABERT_LOCAL_PATH,
        enable_grad_checkpointing=True,
        attn_heads=4,
        attn_dropout=0.1,
    ):
        super().__init__()

        # ---- 1. Load the model configuration ----
        cfg = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,        # allows code from the DNABERT-2 snapshot to be run
            local_files_only=True,         # stops it trying to access the internet (compute nodes don't have internet)
            return_dict=True,              # outputs "ModelOutput" objects instead of plain tuples
            output_hidden_states=True,     # outputs hidden states from every layer
            output_attentions=False        # doesn't output attention matricies (wuold use lots of memory)
        )
        # using flash attention if possible
        for attr in ("use_flash_attn", "flash_attn", "use_flash_attn_mha"):
            if hasattr(cfg, attr):
                setattr(cfg, attr, True)

        # ---- 2. Load the base DNABERT-2 model (weights and model architecture) ----
        self.dnabert = AutoModel.from_pretrained(
            model_name,
            config=cfg,
            trust_remote_code=True,
            local_files_only=True,
        )

        # ---- 3. Define the LoRA configuration ONCE (LoRA is used to update the weights) ----
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # specifies the high-level usage pattern of the base model
            r=16,                         # The rank of the matrices used in the LoRA update, Increased capacity for more effective learning
            lora_alpha=32,                # Standard practice: 2 * r (controls the strength of the update relative to the frozen weights)
            target_modules=[              # modules who's weights get updated
                "Wqkv",                   # Attention: Query, Key, Value
                "dense",                  # Attention: Output projection (after the weights from each head are concattonated together, they go through a linear layer which the "dense" weights refer to)
                "gated_layers",           # FFN: the two input projections for the GEGLU feed-forward block (from the hidden state into the gated activation)
                "wo"                      # FFN: the output projection (from the GEGLU activation back to the model dimension)
            ],
            lora_dropout=0.1,             # Dropout helps prevent overfitting
            bias="none",                  # Doesn't train bias terms
        )

        # ---- 4. Apply PEFT to the base model ONCE ----
        # This function handles freezing the base model and setting up LoRA layers
        self.dnabert = get_peft_model(self.dnabert, peft_config)

        print("\nTrainable parameters after applying LoRA:")
        self.dnabert.print_trainable_parameters()
        print("-" * 40)

        if enable_grad_checkpointing:
            try:
                self.dnabert.gradient_checkpointing_enable()
            except Exception as e:
                print(f"Could not enable gradient checkpointing: {e}")
        
        
        # ---- 5. small chunk-level self-attention ---

        H = self.dnabert.config.hidden_size # dimentions if the hidden state

        # MultiheadAttention expects [B, N, H] if batch_first=True
        self.chunk_attn = nn.MultiheadAttention(
            embed_dim=H,           # embedding dimentions of each chunk
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,      # The first dimention of the input is the number of forkcalls in the batch
        )
        self.chunk_attn_norm = nn.LayerNorm(H)

        # final regressor after pooling across chunks
        self.regressor = nn.Linear(H, 1)

    def forward(self, input_ids, attention_mask, chunk_mask, **kwargs):
        """
        input_ids:      [B, N, L]
        attention_mask: [B, N, L]
        chunk_mask:     [B, N]  True=real chunk, False=padding chunk
        Returns:
          preds: [B]
        """
        B, N, L = input_ids.shape

        # Flatten chunks so DNABERT runs once over all chunks
        flat_input_ids = input_ids.view(B * N, L)
        flat_attn_mask = attention_mask.view(B * N, L)

        # running the flattened inputs through DNABERT-2
        outputs = self.dnabert(
            input_ids=flat_input_ids,
            attention_mask=flat_attn_mask,
            output_hidden_states=False,
            **kwargs
        )


        # extracting the last hidden states 
        last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state # [B*N, L, H]
        # extracting the last hidden state just for the CLS tokens
        cls = last_hidden[:, 0, :]  # [B*N, H] 

        # Reshaping the tensor back to [B, N, H]
        # (splits the first dimention from [B*N] back to [B, N] and keeps the last dimention as H)
        chunk_emb = cls.view(B, N, -1)

        # Key padding mask: True means "ignore" (flips True and False from chunk_mask)
        key_padding_mask = ~chunk_mask  # [B, N]

        # Self-attention over chunks
        attn_out, _ = self.chunk_attn(
            query=chunk_emb,        # Q: what each chunk is "asking" about (who should I attend to?)
            key=chunk_emb,          # K: what each chunk "contains" for matching (used to compute similarity)
            value=chunk_emb,        # V: the actual chunk information that will be mixed/propagated
            key_padding_mask=key_padding_mask,  # mask padded chunks (True = ignore position)
            need_weights=False,     # don't return attention matrices (saves memory)
        )

        chunk_emb2 = self.chunk_attn_norm(chunk_emb + attn_out)

        # Masked mean pooling over chunks -> [B, H]
        # this combines the information from all chunks from the same fork call be taking their mean (not including padding)
        mask_f = chunk_mask.unsqueeze(-1).float()  # [B, N, 1]
        pooled = (chunk_emb2 * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0) # [B, H]

        preds = self.regressor(pooled).squeeze(-1)  # [B]    linear layer [B, H] -> [B, 2] then .squeeze(-1) -> [B]
        return preds















