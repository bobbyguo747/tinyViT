from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


Shape = tuple[int, ...]


def tensor_shape(tensor: torch.Tensor) -> Shape:
    """Return a tensor shape as an ordinary tuple for readable tracing."""
    return tuple(tensor.shape)


class PatchEmbedding(nn.Module):
    """Split an image into non-overlapping patches and embed every patch.

    A Conv2d with kernel_size=stride=patch_size performs two operations at once:
    it cuts the image into patches and linearly projects each patch to embed_dim.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tensor_shape(images)}")
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected {self.image_size}x{self.image_size} images, "
                f"got {images.shape[-2]}x{images.shape[-1]}"
            )

        feature_map = self.projection(images)  # [B, embed_dim, grid, grid]
        tokens = feature_map.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return feature_map, tokens


class MultiHeadSelfAttention(nn.Module):
    """Explicit multi-head self-attention with visible Q/K/V operations."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        # One projection is equivalent to three separate Linear layers, then split.
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self, tokens: torch.Tensor, return_trace: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Shape]]:
        batch_size, token_count, embed_dim = tokens.shape

        qkv = self.qkv(tokens)  # [B, N, 3C]
        query, key, value = qkv.chunk(3, dim=-1)  # three tensors: [B, N, C]

        # [B, N, C] -> [B, N, heads, head_dim] -> [B, heads, N, head_dim]
        query = query.reshape(batch_size, token_count, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, token_count, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, token_count, self.num_heads, self.head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Every token compares with every token inside each attention head.
        attention_scores = (query @ key.transpose(-2, -1)) * self.scale
        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        context = attention_weights @ value  # [B, heads, N, head_dim]
        merged = context.transpose(1, 2).contiguous().reshape(
            batch_size, token_count, embed_dim
        )
        output = self.output_projection(merged)
        output = self.output_dropout(output)

        if not return_trace:
            return output

        trace = {
            "qkv_projection": tensor_shape(qkv),
            "query_per_head": tensor_shape(query),
            "key_per_head": tensor_shape(key),
            "value_per_head": tensor_shape(value),
            "attention_scores": tensor_shape(attention_scores),
            "attention_weights": tensor_shape(attention_weights),
            "merged_heads": tensor_shape(merged),
            "attention_output": tensor_shape(output),
        }
        return output, trace


class FeedForward(nn.Module):
    """Position-wise MLP used inside a Transformer encoder block."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.layers(tokens)


class TransformerEncoderBlock(nn.Module):
    """Pre-LayerNorm Transformer encoder with two residual connections."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=embed_dim * mlp_ratio,
            dropout=dropout,
        )

    def forward(
        self, tokens: torch.Tensor, return_trace: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        normalized_for_attention = self.norm1(tokens)
        if return_trace:
            attention_output, attention_trace = self.attention(
                normalized_for_attention, return_trace=True
            )
        else:
            attention_output = self.attention(normalized_for_attention)
            attention_trace = None

        after_attention_residual = tokens + attention_output
        normalized_for_mlp = self.norm2(after_attention_residual)
        output = after_attention_residual + self.feed_forward(normalized_for_mlp)

        if not return_trace:
            return output

        trace = {
            "input": tensor_shape(tokens),
            "after_norm1": tensor_shape(normalized_for_attention),
            "attention": attention_trace,
            "after_attention_residual": tensor_shape(after_attention_residual),
            "after_norm2": tensor_shape(normalized_for_mlp),
            "output": tensor_shape(output),
        }
        return output, trace


@dataclass(frozen=True)
class TinyViTConfig:
    image_size: int = 28
    in_channels: int = 1
    num_classes: int = 10
    patch_size: int = 4
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    mlp_ratio: int = 4
    dropout: float = 0.1
    use_position_embedding: bool = True


class TinyViT(nn.Module):
    """A small Vision Transformer designed for transparent learning."""

    def __init__(self, config: TinyViTConfig) -> None:
        super().__init__()
        if config.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if config.num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.config = config
        self.patch_embedding = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )
        token_count = self.patch_embedding.num_patches + 1

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        if config.use_position_embedding:
            self.position_embedding = nn.Parameter(
                torch.zeros(1, token_count, config.embed_dim)
            )
        else:
            self.register_parameter("position_embedding", None)

        self.embedding_dropout = nn.Dropout(config.dropout)
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.classifier = nn.Linear(config.embed_dim, config.num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.position_embedding is not None:
            nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(
        self, images: torch.Tensor, return_trace: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        patch_feature_map, patch_tokens = self.patch_embedding(images)
        batch_size = images.shape[0]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, patch_tokens), dim=1)
        tokens_with_cls_shape = tensor_shape(tokens)

        if self.position_embedding is not None:
            tokens = tokens + self.position_embedding
        tokens = self.embedding_dropout(tokens)

        encoder_traces: list[dict[str, Any]] = []
        for block in self.encoder_blocks:
            if return_trace:
                tokens, block_trace = block(tokens, return_trace=True)
                encoder_traces.append(block_trace)
            else:
                tokens = block(tokens)

        encoder_output = self.final_norm(tokens)
        cls_vector = encoder_output[:, 0]
        logits = self.classifier(cls_vector)  # raw logits; no classifier Softmax

        if not return_trace:
            return logits

        trace = {
            "input": tensor_shape(images),
            "patch_feature_map": tensor_shape(patch_feature_map),
            "patch_tokens": tensor_shape(patch_tokens),
            "tokens_with_cls": tokens_with_cls_shape,
            "position_embedding": (
                tensor_shape(self.position_embedding)
                if self.position_embedding is not None
                else None
            ),
            "tokens_entering_encoder": tensor_shape(tokens)
            if not encoder_traces
            else encoder_traces[0]["input"],
            "encoder_blocks": encoder_traces,
            "encoder_output": tensor_shape(encoder_output),
            "cls_vector": tensor_shape(cls_vector),
            "logits": tensor_shape(logits),
        }
        return logits, trace


def build_tinyvit(
    model_config: dict[str, Any], use_position_embedding: bool = True
) -> TinyViT:
    """Build TinyViT from the model section of config.yaml."""
    config = TinyViTConfig(
        image_size=int(model_config["image_size"]),
        in_channels=int(model_config["in_channels"]),
        num_classes=int(model_config["num_classes"]),
        patch_size=int(model_config["patch_size"]),
        embed_dim=int(model_config["embed_dim"]),
        num_heads=int(model_config["num_heads"]),
        num_layers=int(model_config["num_layers"]),
        mlp_ratio=int(model_config["mlp_ratio"]),
        dropout=float(model_config["dropout"]),
        use_position_embedding=use_position_embedding,
    )
    return TinyViT(config)


@dataclass(frozen=True)
class SimpleCNNConfig:
    image_size: int = 28
    in_channels: int = 1
    num_classes: int = 10
    dropout: float = 0.1


class SimpleCNN(nn.Module):
    """A compact convolutional baseline with two easy-to-explain blocks."""

    def __init__(self, config: SimpleCNNConfig) -> None:
        super().__init__()
        if config.image_size % 4 != 0:
            raise ValueError("SimpleCNN image_size must be divisible by 4")
        self.config = config
        self.block1 = nn.Sequential(
            nn.Conv2d(config.in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        final_grid = config.image_size // 4
        flattened_features = 64 * final_grid * final_grid
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(flattened_features, config.num_classes)

    def forward(
        self, images: torch.Tensor, return_trace: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Shape]]:
        after_block1 = self.block1(images)  # [B, 32, 14, 14]
        after_block2 = self.block2(after_block1)  # [B, 64, 7, 7]
        flattened = after_block2.flatten(1)  # [B, 64*7*7]
        logits = self.classifier(self.dropout(flattened))  # raw logits

        if not return_trace:
            return logits
        return logits, {
            "input": tensor_shape(images),
            "after_block1": tensor_shape(after_block1),
            "after_block2": tensor_shape(after_block2),
            "flattened": tensor_shape(flattened),
            "logits": tensor_shape(logits),
        }


MODEL_NAMES = ("tinyvit", "tinyvit_no_pos", "cnn")


def build_model(model_name: str, model_config: dict[str, Any]) -> nn.Module:
    """Build one of the three pre-declared experiments from config.yaml."""
    if model_name == "tinyvit":
        return build_tinyvit(model_config, use_position_embedding=True)
    if model_name == "tinyvit_no_pos":
        return build_tinyvit(model_config, use_position_embedding=False)
    if model_name == "cnn":
        return SimpleCNN(
            SimpleCNNConfig(
                image_size=int(model_config["image_size"]),
                in_channels=int(model_config["in_channels"]),
                num_classes=int(model_config["num_classes"]),
                dropout=float(model_config["dropout"]),
            )
        )
    raise ValueError(f"Unknown model_name={model_name!r}; choose from {MODEL_NAMES}")


def build_model_from_saved_config(
    model_name: str, saved_config: dict[str, Any]
) -> nn.Module:
    """Rebuild the exact architecture recorded inside a checkpoint."""
    if model_name in ("tinyvit", "tinyvit_no_pos"):
        return TinyViT(TinyViTConfig(**saved_config))
    if model_name == "cnn":
        return SimpleCNN(SimpleCNNConfig(**saved_config))
    raise ValueError(f"Unsupported checkpoint model_name={model_name!r}")
