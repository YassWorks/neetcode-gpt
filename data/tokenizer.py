from typing import List
from collections import defaultdict


class Solution:

    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        merges = []
        token_list = [c for c in corpus]
        for _ in range(num_merges):
            freqs = defaultdict(int)
            for i in range(len(token_list)-1):
                tup = (token_list[i], token_list[i+1])
                freqs[tup] += 1
            highest = max(freqs.values())
            candidates = sorted([k for k,v in freqs.items() if v==highest])
            target = candidates[0]
            new_token = target[0]+target[1]
            merges.append(list(target))
            i = 0
            while (i < len(token_list)-1):
                if token_list[i]==target[0] and token_list[i+1]==target[1]:
                    token_list[i]=new_token
                    token_list.pop(i+1)
                    i -= 1
                i += 1
        return merges
