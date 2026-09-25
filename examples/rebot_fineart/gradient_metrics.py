"""Inspect gradients without overflowing their low-precision norm reduction."""

import math

import torch


def json_number(value: float) -> float | str:
    return value if math.isfinite(value) else str(value)


def gradient_summary(named_parameters) -> dict:
    rows = []
    missing = 0
    for name, parameter in named_parameters:
        if parameter.grad is None:
            missing += int(parameter.requires_grad)
            continue
        grad = parameter.grad.detach()
        nonfinite = int((~torch.isfinite(grad)).sum().item())
        norm = torch.linalg.vector_norm(grad, dtype=torch.float64).item()
        rows.append(
            {
                "parameter": name,
                "dtype": str(grad.dtype),
                "elements": grad.numel(),
                "nonfinite_elements": nonfinite,
                "norm_float64": json_number(norm),
                "max_abs": json_number(grad.abs().max().item()),
            }
        )
    if not rows:
        raise ValueError("No gradients were produced")
    total = math.hypot(*(float(row["norm_float64"]) for row in rows))
    ordered = sorted(
        rows,
        key=lambda row: float("inf") if row["nonfinite_elements"] else float(row["norm_float64"]),
        reverse=True,
    )
    return {
        "norm_float64": json_number(total),
        "parameters_with_grad": len(rows),
        "trainable_parameters_without_grad": missing,
        "nonfinite_elements": sum(row["nonfinite_elements"] for row in rows),
        "nonfinite_parameters": [row for row in rows if row["nonfinite_elements"]],
        "largest_gradients": ordered[:20],
    }
