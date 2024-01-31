import torch

from src.third_party.open_clip.clip import tokenize


# tokenize the prompt


def process_prompt(prompt):
    text = []

    for p in prompt:
        text_tokens = tokenize(p) if "*" in p else tokenize(f"a photo of * , {p}")
        text.append(text_tokens)
    text = torch.cat(text, dim=0)
    return text


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)
