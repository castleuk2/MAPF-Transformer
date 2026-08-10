import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .structured_map import ModelConfig as MapConfig
from .structured_map import StructuredMapTransformer
from .continuous_tokenizer import EncodedObservation


@dataclass
class GPTConfig:
    block_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False


class LayerNorm(nn.Module):
    def __init__(self, size: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class NonCausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.n_head, self.n_embd, self.dropout = cfg.n_head, cfg.n_embd, cfg.dropout

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        shape = (b, t, self.n_head, c // self.n_head)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.attn = NonCausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd, cfg.bias)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias), nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MAPFGPT(nn.Module):
    """Vocabulary-free tokens followed by the original MAPF-GPT Transformer."""
    def __init__(self, cfg: GPTConfig, map_config: MapConfig | None = None):
        super().__init__()
        self.cfg = cfg
        if cfg.block_size != 256:
            raise ValueError("Strict Semantic layout requires block_size=256")
        if map_config is None:
            map_config = MapConfig(
                d_model=cfg.n_embd, num_heads=cfg.n_head, dropout=cfg.dropout,
                reconstruction_classes=2,
            )
        if map_config.d_model != cfg.n_embd:
            raise ValueError("map d_model must equal the MAPF-GPT embedding dimension")
        self.feature_encoder = nn.Sequential(
            nn.Linear(16, cfg.n_embd), nn.GELU(),
            nn.Linear(cfg.n_embd, cfg.n_embd), LayerNorm(cfg.n_embd, cfg.bias),
        )
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.agent_slot_embedding = nn.Embedding(14, cfg.n_embd)
        self.field_embedding = nn.Embedding(11, cfg.n_embd)
        self.lag_embedding = nn.Embedding(7, cfg.n_embd)
        self.role_embedding = nn.Embedding(3, cfg.n_embd)
        self.valid_embedding = nn.Embedding(2, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = LayerNorm(cfg.n_embd, cfg.bias)
        self.action_head = nn.Linear(cfg.n_embd, 5)
        self.apply(self._init)
        for name, parameter in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(parameter, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))
        # Construct after MAPF-GPT initialization so the structured encoder's
        # own Xavier/relative-bias initialization is not overwritten.
        self.map_encoder = StructuredMapTransformer(map_config)
        self._map_encoder_frozen = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self._map_encoder_frozen:
            # Freezing parameters alone does not disable dropout. Keep the
            # pretrained map backbone deterministic during policy training.
            self.map_encoder.eval()
        return self

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, semantics: EncodedObservation, targets=None, *, halo_maps=None):
        b, t, feature_dim = semantics.token_features.shape
        if t != self.cfg.block_size:
            raise ValueError(f"expected exactly {self.cfg.block_size} tokens, got {t}")
        if feature_dim != 16: raise ValueError("expected 16 feature channels")
        pos = torch.arange(t, device=semantics.token_features.device)
        token_embeddings = self.feature_encoder(semantics.token_features) + (
            self.agent_slot_embedding(semantics.agent_slots)
            + self.field_embedding(semantics.field_ids)
            + self.lag_embedding(semantics.lag_ids)
            + self.role_embedding(semantics.role_ids)
            + self.valid_embedding(semantics.valid_ids)
        )
        if halo_maps is not None:
            map_tokens = self.map_encoder(halo_maps).latent_tokens
            if map_tokens.shape != (b, 25, self.cfg.n_embd):
                raise ValueError(f"map encoder returned unexpected shape {tuple(map_tokens.shape)}")
            # Slots 0..24 are the exact continuous tokens produced by the
            # structured map encoder. Absolute MAPF-GPT positions are then
            # added uniformly to all 256 slots.
            token_embeddings = torch.cat((map_tokens, token_embeddings[:, 25:]), dim=1)
        x = self.drop(token_embeddings + self.position_embedding(pos))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.action_head(x[:, -1, :])
        loss = F.cross_entropy(logits, targets) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def action_distribution(self, observation, *, halo_maps=None):
        logits, _ = self(observation, halo_maps=halo_maps)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def act(self, observation, sample=True, generator=None, *, halo_maps=None):
        probs = self.action_distribution(observation, halo_maps=halo_maps)
        if sample:
            return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        return probs.argmax(dim=-1)

    def load_map_encoder(self, checkpoint_path, *, freeze=False, strict=True):
        """Load either a raw map state_dict or the project's checkpoint wrapper."""
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload.get("model_state", payload.get("model_state_dict", payload))
        result = self.map_encoder.load_state_dict(state, strict=strict)
        if freeze:
            self.freeze_map_encoder()
        return result

    def freeze_map_encoder(self):
        self.map_encoder.requires_grad_(False)
        self.map_encoder.eval()
        self._map_encoder_frozen = True
