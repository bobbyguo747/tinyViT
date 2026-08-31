from __future__ import annotations

import argparse
from typing import Any

import torch

from model import TinyViT, build_tinyvit
from utils import load_config, select_device, set_seed


def symbolic_shape(shape: tuple[int, ...]) -> str:
    """Render the first (batch) dimension as B."""
    dimensions = ["B", *[str(value) for value in shape[1:]]]
    return "[" + ", ".join(dimensions) + "]"


def literal_shape(shape: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(value) for value in shape) + "]"


def print_attention_trace(trace: dict[str, Any]) -> None:
    attention = trace["encoder_blocks"][0]["attention"]
    print("\n第 1 个 Encoder 内部的注意力 shape：")
    print(f"  QKV 一次投影:       {symbolic_shape(attention['qkv_projection'])}")
    print(f"  Query 分头后:       {symbolic_shape(attention['query_per_head'])}")
    print(f"  Key 分头后:         {symbolic_shape(attention['key_per_head'])}")
    print(f"  Value 分头后:       {symbolic_shape(attention['value_per_head'])}")
    print(f"  注意力分数 QK^T:    {symbolic_shape(attention['attention_scores'])}")
    print(f"  Softmax 权重:        {symbolic_shape(attention['attention_weights'])}")
    print(f"  多头合并:            {symbolic_shape(attention['merged_heads'])}")


def inspect_model(
    model: TinyViT, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model = model.to(device)
    model.eval()
    config = model.config
    images = torch.zeros(
        batch_size,
        config.in_channels,
        config.image_size,
        config.image_size,
        device=device,
    )

    with torch.no_grad():
        logits, trace = model(images, return_trace=True)

    grid_size = config.image_size // config.patch_size
    patch_count = grid_size**2
    token_count = patch_count + 1
    expected_shapes = {
        "input": (
            batch_size,
            config.in_channels,
            config.image_size,
            config.image_size,
        ),
        "patch_feature_map": (batch_size, config.embed_dim, grid_size, grid_size),
        "patch_tokens": (batch_size, patch_count, config.embed_dim),
        "tokens_with_cls": (batch_size, token_count, config.embed_dim),
        "encoder_output": (batch_size, token_count, config.embed_dim),
        "cls_vector": (batch_size, config.embed_dim),
        "logits": (batch_size, config.num_classes),
    }
    for name, expected in expected_shapes.items():
        actual = trace[name]
        if actual != expected:
            raise AssertionError(f"{name}: expected {expected}, got {actual}")

    print(f"device: {device}")
    print(f"位置编码: {'启用' if model.position_embedding is not None else '禁用'}")
    print(f"参数量: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print("\nTinyViT 主数据流：")
    print(f"  输入:               {symbolic_shape(trace['input'])}")
    print(f"  Patch Embedding:    {symbolic_shape(trace['patch_feature_map'])}")
    print(f"  Patch Tokens:       {symbolic_shape(trace['patch_tokens'])}")
    print(f"  加入 CLS:           {symbolic_shape(trace['tokens_with_cls'])}")
    if trace["position_embedding"] is not None:
        print(f"  位置编码参数:        {literal_shape(trace['position_embedding'])}")
    else:
        print("  位置编码参数:        None（消融设置）")
    for index, block_trace in enumerate(trace["encoder_blocks"], start=1):
        print(f"  Encoder Block {index}: {symbolic_shape(block_trace['output'])}")
    print(f"  Encoder 输出:       {symbolic_shape(trace['encoder_output'])}")
    print(f"  CLS 向量:           {symbolic_shape(trace['cls_vector'])}")
    print(f"  分类 logits:        {symbolic_shape(trace['logits'])}")
    print_attention_trace(trace)
    print("\nshape 检查通过。分类器输出为原始 logits，没有手动 Softmax。")
    return logits, trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect every important TinyViT shape")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--no-position-encoding",
        action="store_true",
        help="Build the position-encoding ablation model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    selected_device = select_device(config.get("device", "auto"))
    tinyvit = build_tinyvit(
        config["model"],
        use_position_embedding=not args.no_position_encoding,
    )
    inspect_model(tinyvit, args.batch_size, selected_device)
