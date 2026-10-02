"""Generate report figures from verified experiment JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load(name: str) -> dict:
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def confusion_matrix(model: str, payload: dict) -> None:
    image = Image.new("RGB", (1000, 430), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=16)
    draw.text((24, 18), f"{model} confusion matrices", fill="black", font=font)
    for panel, split in enumerate(("validation", "test")):
        matrix = payload[split]["confusion_matrix"]
        left = 90 + panel * 430
        top = 95
        draw.text((left, 58), split.title(), fill="black", font=font)
        maximum = max(max(row) for row in matrix)
        for row in range(2):
            for column in range(2):
                value = matrix[row][column]
                intensity = int(245 - 175 * value / maximum)
                box = (left + column * 135, top + row * 115, left + (column + 1) * 135, top + (row + 1) * 115)
                draw.rectangle(box, fill=(intensity, intensity + 5, 255), outline="black")
                draw.text((box[0] + 55, box[1] + 45), str(value), fill="black", font=font)
        draw.text((left + 43, 325), "predicted", fill="black", font=small)
        draw.text((left - 68, 190), "true", fill="black", font=small)
        draw.text((left + 25, 365), "negative       positive", fill="black", font=small)
    image.save(OUT / f"{model.lower()}_confusion_matrices.png")


def comparison(payloads: list[dict]) -> None:
    image = Image.new("RGB", (1000, 560), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=16)
    draw.text((28, 20), "Verified interim test results", fill="black", font=font)
    chart = (100, 80, 930, 440)
    draw.rectangle(chart, outline="black")
    for tick in range(0, 101, 20):
        y = chart[3] - int((chart[3] - chart[1]) * tick / 100)
        draw.line((chart[0], y, chart[2], y), fill=(220, 220, 220))
        draw.text((55, y - 8), str(tick), fill="black", font=small)
    for index, item in enumerate(payloads):
        center = 225 + index * 250
        values = (item["test"]["accuracy"] * 100, item["test"]["f1"] * 100)
        for offset, value, color, label in ((-38, values[0], (46, 110, 180), "Acc"), (38, values[1], (220, 120, 45), "F1")):
            height = int((chart[3] - chart[1]) * value / 100)
            box = (center + offset - 22, chart[3] - height, center + offset + 22, chart[3])
            draw.rectangle(box, fill=color)
            draw.text((box[0] - 4, box[1] - 22), f"{value:.1f}", fill="black", font=small)
            draw.text((box[0] + 2, chart[3] + 12), label, fill="black", font=small)
        draw.text((center - 32, chart[3] + 42), item["model"], fill="black", font=font)
    draw.text((640, 25), "blue = accuracy   orange = F1", fill="black", font=small)
    image.save(OUT / "verified_test_comparison.png")


def main() -> None:
    payloads = [load("mlp_metrics.json"), load("cnn_metrics.json"), load("bilstm_metrics.json")]
    for payload in payloads:
        confusion_matrix(payload["model"], payload)
    comparison(payloads)


if __name__ == "__main__":
    main()
