import numpy as np
import random
from itertools import product
from math import comb
from typing import Any
from exceptions import CodeParametrsError


class TernaryCode:
    def __init__(self, length: int, dim: int) -> None:
        self.length = length
        self.dim = dim
        self.min_dist = length - dim + 1

        if not self._validate_params(length, dim, self.min_dist):
            raise CodeParametrsError("Неверные параметры кода")

        self.gen_matrix, self.check_matrix, self.actual_dist = self._build_code(
            dim, length, self.min_dist
        )
        self.codeword_count = 3**dim
        self.max_errors = (self.actual_dist - 1) // 2

    def _build_code(self, k: int, n: int, d: int, p: int = 3):
        while True:
            rand_part = np.random.randint(0, p, size=(k, n - k))
            id_mat = np.eye(k, dtype=int)
            full_matrix = np.hstack((id_mat, rand_part))

            if np.linalg.matrix_rank(full_matrix) == k:
                code_dist = self._calc_code_distance(full_matrix, p)
                break

        id_check = np.eye(n - k, dtype=int)
        parity_matrix = np.hstack((-rand_part.T % p, id_check))
        return full_matrix, parity_matrix, code_dist

    def _calc_code_distance(self, matrix: Any, p: int = 3) -> int:
        min_wt = self.length
        for msg in product(range(p), repeat=self.dim):
            cw = np.dot(msg, matrix) % p
            wt = np.count_nonzero(cw)
            if 0 < wt < min_wt:
                min_wt = wt
        return min_wt

    def _gilbert_bound(self, n: int, k: int, d: int) -> float:
        vol = sum(comb(n, i) * (2**i) for i in range(d - 1))
        return 3**k >= 3**n / vol

    def _hamming_bound(self, n: int, k: int, d: int) -> float:
        vol = sum(comb(n, i) * (2**i) for i in range((d - 1) // 2 + 1))
        return 3**k <= 3**n / vol

    def _singlton_bound(self, n: int, k: int, d: int) -> float:
        return k <= n - d + 1

    def _validate_params(self, n: int, k: int, d: int):
        if not self._hamming_bound(n, k, d):
            return False
        if not self._singlton_bound(n, k, d):
            return False
        if not self._gilbert_bound(n, k, d):
            return False
        return True

    def decode(self, received: Any):
        k, n = self.gen_matrix.shape
        code_map = {
            tuple(np.dot(m, self.gen_matrix) % 3): m
            for m in product(range(3), repeat=k)
        }

        best_dist = float("inf")
        best_msg = None

        for _ in range(100):
            sample_pos = random.sample(range(n), k)
            candidates = [
                cw for cw in code_map if all(cw[i] == received[i] for i in sample_pos)
            ]

            for cw in candidates:
                dist = np.sum(np.array(cw) != received)
                if dist < best_dist:
                    best_msg = code_map[cw]
                    best_dist = dist

        return best_msg, best_dist

    def encode(self, data: Any) -> int:
        return np.dot(data, self.gen_matrix) % 3


if __name__ == "__main__":
    try:
        code = TernaryCode(5, 3)
    except CodeParametrsError as e:
        print(e)
