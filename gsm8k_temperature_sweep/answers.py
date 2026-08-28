"""GSM8K answer extraction without model-runtime dependencies."""
from __future__ import annotations

import re


def normalize_number(text: str) -> str:
    return text.strip().replace(",", "").strip(" .:$").lower()


def boxed_payloads(text: str) -> list[str]:
    payloads: list[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth = 1
        start = match.end()
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    payloads.append(text[start:index])
                    break
    return payloads


def extract_gsm8k_answer(text: str) -> str:
    candidates = boxed_payloads(text)
    if candidates:
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", candidates[-1])
        if numbers:
            return normalize_number(numbers[-1])
    marker = re.search(r"####\s*([^\n]+)", text)
    if marker:
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", marker.group(1))
        if numbers:
            return normalize_number(numbers[0])
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return normalize_number(numbers[-1]) if numbers else ""
