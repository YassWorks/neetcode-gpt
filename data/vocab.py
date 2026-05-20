from typing import Dict, List, Tuple


class Solution:

    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        vocab = list(text)
        vocab = sorted(list(set(vocab)))
        stoi = {c: i for c, i in enumerate(vocab)}
        itos = {i: c for c, i in enumerate(vocab)}
        return (itos, stoi)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        text_list = list(text)
        result = []
        for i in text_list:
            result.append(stoi[i])
        return result

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        result = []
        for idx in ids:
            result.append(itos[idx])
        return ''.join(result)
