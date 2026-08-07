"""
Phase 0 — Environment check.
Run: python 00_check_setup.py
Confirms all packages import correctly and checks whether PyTorch
can use your M2's GPU (MPS backend) for the CVAE training in Phase 5.
"""

import sys

def check(name, import_stmt):
    try:
        exec(import_stmt)
        print(f"[OK]   {name}")
        return True
    except Exception as e:
        print(f"[FAIL] {name} -> {e}")
        return False

ok = True
ok &= check("pandas", "import pandas")
ok &= check("numpy", "import numpy")
ok &= check("scikit-learn", "import sklearn")
ok &= check("causal-learn", "from causallearn.search.ConstraintBased.PC import pc")
ok &= check("networkx", "import networkx")
ok &= check("matplotlib", "import matplotlib")
ok &= check("torch", "import torch")

print()
if ok:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    if torch.backends.mps.is_available():
        print("[OK]   MPS (Apple Silicon GPU) is available -> CVAE training will use device='mps'")
    else:
        print("[INFO] MPS not available -> CVAE training will fall back to device='cpu' (still fine, just slower)")
    print("\nAll checks passed. You're ready for Phase 2 (data).")
else:
    print("\nSome packages failed to import. Re-run:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
