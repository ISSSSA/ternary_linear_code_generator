import numpy as np
from itertools import product
from math import comb
import random
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox


class TernaryCodeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ternary Code Encoder/Decoder")
        self.style = ttk.Style(theme='darkly')
        self.setup_ui()

        # Initialize code variables
        self.code = None
        self.encoded_msg = None

    def setup_ui(self):
        # Main container
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=BOTH, expand=YES)

        # Code parameters frame
        param_frame = ttk.Labelframe(self.main_frame, text="Code Parameters", padding=10)
        param_frame.pack(fill=X, pady=5)

        ttk.Label(param_frame, text="Length (n):").grid(row=0, column=0, padx=5, pady=5, sticky=W)
        self.n_entry = ttk.Entry(param_frame)
        self.n_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(param_frame, text="Dimension (k):").grid(row=1, column=0, padx=5, pady=5, sticky=W)
        self.k_entry = ttk.Entry(param_frame)
        self.k_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(
            param_frame,
            text="Generate Code",
            command=self.generate_code,
            bootstyle=(SUCCESS, OUTLINE)
        ).grid(row=2, column=0, columnspan=2, pady=5)

        # Code info frame
        self.info_frame = ttk.Labelframe(self.main_frame, text="Code Information", padding=5)
        self.info_frame.pack(fill=X, pady=5)

        # Encoding frame
        self.encode_frame = ttk.Labelframe(self.main_frame, text="Encoding", padding=5)
        self.encode_frame.pack(fill=X, pady=5)

        ttk.Label(self.encode_frame, text="Message (k digits 0-2):").pack(anchor=W, padx=5, pady=2)
        self.msg_entry = ttk.Entry(self.encode_frame)
        self.msg_entry.pack(fill=X, padx=5, pady=5)

        ttk.Button(
            self.encode_frame,
            text="Encode Message",
            command=self.encode_message,
            bootstyle=(SUCCESS, OUTLINE)
        ).pack(pady=5)

        self.encoded_label = ttk.Label(self.encode_frame, text="Encoded message: ")
        self.encoded_label.pack(anchor=W, padx=5, pady=2)

        # Error frame
        self.error_frame = ttk.Labelframe(self.main_frame, text="Add Error", padding=5)
        self.error_frame.pack(fill=X, pady=5)

        ttk.Label(self.error_frame, text="Error vector (n digits 0-2):").pack(anchor=W, padx=5, pady=2)
        self.error_entry = ttk.Entry(self.error_frame)
        self.error_entry.pack(fill=X, padx=5, pady=5)

        ttk.Button(
            self.error_frame,
            text="Add Error and Decode",
            command=self.add_error_and_decode,
            bootstyle=(DANGER, OUTLINE)
        ).pack(pady=5)

        # Results frame
        self.results_frame = ttk.Labelframe(self.main_frame, text="Decoding Results", padding=5)
        self.results_frame.pack(fill=X, pady=5)

        self.received_label = ttk.Label(self.results_frame, text="Received message: ")
        self.received_label.pack(anchor=W, padx=5, pady=2)

        self.decoded_label = ttk.Label(self.results_frame, text="Decoded message: ")
        self.decoded_label.pack(anchor=W, padx=5, pady=2)

        self.errors_label = ttk.Label(self.results_frame, text="Errors detected: ")
        self.errors_label.pack(anchor=W, padx=5, pady=2)

        self.success_label = ttk.Label(self.results_frame, text="Decoding success: ")
        self.success_label.pack(anchor=W, padx=5, pady=2)

    def generate_code(self):
        try:
            n = int(self.n_entry.get())
            k = int(self.k_entry.get())

            if n <= 0 or k <= 0:
                raise ValueError("Parameters must be positive")
            if k >= n:
                raise ValueError("Dimension (k) must be less than length (n)")

            self.code = TernaryCode(n, k)
            self.show_code_info()

            # Enable other sections
            self.msg_entry.config(state=NORMAL)
            self.error_entry.config(state=NORMAL)

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid parameters: {str(e)}")

    def show_code_info(self):
        # Clear previous info
        for widget in self.info_frame.winfo_children():
            widget.destroy()

        ttk.Label(self.info_frame, text=f"Code length (n): {self.code.length}").pack(anchor=W)
        ttk.Label(self.info_frame, text=f"Code dimension (k): {self.code.dim}").pack(anchor=W)
        ttk.Label(self.info_frame, text=f"Minimum distance: {self.code.actual_dist}").pack(anchor=W)
        ttk.Label(self.info_frame, text=f"Correctable errors: {self.code.max_errors}").pack(anchor=W)

        # Show matrices in scrollable text widgets
        ttk.Label(self.info_frame, text="Generator matrix:").pack(anchor=W, pady=(10, 0))
        gen_text = ttk.Text(self.info_frame, height=4, width=50)
        gen_text.pack(fill=X, padx=5)
        gen_text.insert(END, str(self.code.gen_matrix))
        gen_text.config(state=DISABLED)

        ttk.Label(self.info_frame, text="Parity-check matrix:").pack(anchor=W, pady=(5, 0))
        check_text = ttk.Text(self.info_frame, height=4, width=40)
        check_text.pack(fill=X, padx=5)
        check_text.insert(END, str(self.code.check_matrix))
        check_text.config(state=DISABLED)

    def encode_message(self):
        if not self.code:
            messagebox.showerror("Error", "Please generate code first")
            return

        msg_str = self.msg_entry.get().strip()
        if not msg_str:
            messagebox.showerror("Error", "Please enter a message")
            return

        try:
            msg = np.array([int(c) for c in msg_str.split()])
            if len(msg) != self.code.dim:
                raise ValueError(f"Message must have {self.code.dim} digits")
            if any(x not in {0, 1, 2} for x in msg):
                raise ValueError("Only 0, 1, 2 are allowed")

            self.encoded_msg = self.code.encode(msg)
            self.encoded_label.config(text=f"Encoded message: {' '.join(map(str, self.encoded_msg))}")

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

    def add_error_and_decode(self):
        if not self.encoded_msg.any():
            messagebox.showerror("Error", "Please encode a message first")
            return

        error_str = self.error_entry.get().strip()
        if not error_str:
            messagebox.showerror("Error", "Please enter an error vector")
            return

        try:
            error = np.array([int(c) for c in error_str.split()])
            if len(error) != self.code.length:
                raise ValueError(f"Error vector must have {self.code.length} digits")
            if any(x not in {0, 1, 2} for x in error):
                raise ValueError("Only 0, 1, 2 are allowed")

            # Add error
            received = (self.encoded_msg + error) % 3
            self.received_label.config(text=f"Received message: {' '.join(map(str, received))}")

            # Decode
            decoded, errs = self.code.decode(received)
            original_msg = self.msg_entry.get().strip().split()

            self.decoded_label.config(text=f"Decoded message: {' '.join(map(str, decoded))}")
            self.errors_label.config(text=f"Errors detected: {errs}")

            success = np.array_equal(decoded, [int(x) for x in original_msg])
            self.success_label.config(
                text=f"Decoding success: {'Yes' if success else 'No'}",
                bootstyle=(SUCCESS if success else DANGER)
            )

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))


class TernaryCode:
    def __init__(self, length, dim):
        self.length = length
        self.dim = dim
        self.min_dist = length - dim + 1

        if not self._validate_params(length, dim, self.min_dist):
            raise ValueError("Invalid code parameters")

        self.gen_matrix, self.check_matrix, self.actual_dist = self._build_code(dim, length, self.min_dist)
        self.codeword_count = 3 ** dim
        self.max_errors = (self.actual_dist - 1) // 2

    def _build_code(self, k, n, d, p=3):
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

    def _calc_code_distance(self, matrix, p=3):
        min_wt = self.length
        for msg in product(range(p), repeat=self.dim):
            cw = np.dot(msg, matrix) % p
            wt = np.count_nonzero(cw)
            if 0 < wt < min_wt:
                min_wt = wt
        return min_wt

    def _gilbert_bound_ok(self, n, k, d):
        vol = sum(comb(n, i) * (2 ** i) for i in range(d - 1))
        return 3 ** k >= 3 ** n / vol

    def _hamming_bound_ok(self, n, k, d):
        vol = sum(comb(n, i) * (2 ** i) for i in range((d - 1) // 2 + 1))
        return 3 ** k <= 3 ** n / vol

    def _singleton_bound_ok(self, n, k, d):
        return k <= n - d + 1

    def _validate_params(self, n, k, d):
        if not self._hamming_bound_ok(n, k, d):
            return False
        if not self._singleton_bound_ok(n, k, d):
            return False
        if not self._gilbert_bound_ok(n, k, d):
            return False
        return True

    def decode(self, received):
        k, n = self.gen_matrix.shape
        code_map = {tuple(np.dot(m, self.gen_matrix) % 3): m for m in product(range(3), repeat=k)}

        best_dist = float('inf')
        best_msg = None

        for _ in range(100):
            sample_pos = random.sample(range(n), k)
            candidates = [cw for cw in code_map if all(cw[i] == received[i] for i in sample_pos)]

            for cw in candidates:
                dist = np.sum(np.array(cw) != received)
                if dist < best_dist:
                    best_msg = code_map[cw]
                    best_dist = dist

        return best_msg, best_dist

    def encode(self, data):
        return np.dot(data, self.gen_matrix) % 3


if __name__ == "__main__":
    root = ttk.Window(themename='darkly')
    app = TernaryCodeGUI(root)
    root.mainloop()