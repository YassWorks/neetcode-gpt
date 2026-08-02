from typing import List, Dict


class Solution:
    def _greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            longest = None
            for j in range(i + 1, n + 1):
                sub = text[i:j]
                if sub in vocab:
                    longest = sub

            if longest:
                tokens.append(longest)
                i += len(longest)
            else:
                tokens.append(text[i])
                i += 1
        return tokens

    def tokenize_numbers(
        self, numbers: List[int], vocab: Dict[str, int]
    ) -> List[List[str]]:
        result = []
        for number in numbers:
            result.append(self._greedy_tokenize(str(number), vocab))
        return result

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        return len(self._greedy_tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        words = text.split()
        if not words:
            return 0.0
        token_count = self.count_tokens(text, vocab)
        word_count = len(words)
        return round(token_count / word_count, 4)
