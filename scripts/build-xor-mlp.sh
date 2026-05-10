#!/usr/bin/env bash
# Train a tiny 2-layer XOR MLP in PyTorch, script it via torch.jit, and
# save to models/xor-mlp.pt for the wasi-nn-torch harness.
#
# Requires: python3 with torch installed (`pip install torch`).
#
# Reproducible — fixed seed, deterministic optimizer, ~1 second runtime.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models

OUT="models/xor-mlp.pt"

if [ -f "$OUT" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "✓ $OUT already present (FORCE=1 to rebuild)"
    ls -lh "$OUT"
    exit 0
fi

python3 - "$OUT" <<'PY'
import sys
import torch
import torch.nn as nn

OUT = sys.argv[1]

torch.manual_seed(42)

class XorMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

model = XorMlp()
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

opt = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.BCELoss()

# 2k epochs is overkill for XOR but trivial; keeps the run reproducible
# across hardware that has different fast-paths in BLAS.
for epoch in range(2000):
    pred = model(X)
    loss = loss_fn(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

model.eval()
with torch.no_grad():
    final = model(X).flatten().tolist()

print("training complete; final outputs (sigmoid):")
for (a, b), p in zip([(0, 0), (0, 1), (1, 0), (1, 1)], final):
    pred = 1 if p >= 0.5 else 0
    expected = a ^ b
    mark = "OK" if pred == expected else "FAIL"
    print(f"  XOR({a}, {b}) -> {p:.4f}  pred={pred}  expected={expected}  {mark}")

example = torch.zeros(1, 2)
traced = torch.jit.trace(model, example)
traced.save(OUT)
print(f"\n✓ saved TorchScript module to {OUT}")
PY

echo
ls -lh "$OUT"
echo
echo "done. ready to run:"
echo "  export WACS_WASINN_TORCH_DIR=\$(pwd)/models"
echo "  wacs run target/wasm32-wasip2/release/wasi-nn-torch.wasm \\"
echo "      --wasip2 --bind <path-to-Wacs.WASI.NN.TorchSharp.dll>"
echo
echo "guest's load-by-name target:  xor-mlp"
