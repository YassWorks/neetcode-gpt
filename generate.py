import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution:
    def generate(
        self,
        model,
        new_chars: int,
        context: TensorType[int],
        context_length: int,
        int_to_char: dict,
    ) -> str:
        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        output = ""

        for i in range(new_chars):
            context_crop = context[:, -context_length:]
            logits = model(context_crop)
            logits = logits[:, -1, :]
            dist = torch.softmax(logits, dim=-1)
            generator.set_state(initial_state)
            next_token = torch.multinomial(dist, 1, generator=generator)
            context = torch.cat([context, next_token], dim=1)
            output += int_to_char[next_token.item()]

        return output
