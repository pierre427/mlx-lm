#!/bin/bash
# Cross-Mac validation of mlx-lm PR #1456 (consolidated: GDN exact-replay rollback
# + MTP loading + RotatingKVCache rollback). Self-contained — no patch file needed:
# fetches the PR head directly from GitHub.
#
#   ./validate_pr1456.sh            # unit tests (tiny models, ~3 min)
#   ./validate_pr1456.sh --e2e      # + real weights: Qwen3.5-9B + 0.8B hybrid pair
#                                   #   (~5.5GB download; cached if run before)
set -euo pipefail

echo "== machine =="
sysctl -n machdep.cpu.brand_string || true
sysctl -n hw.memsize | awk '{printf "RAM: %.0f GB\n", $1/1e9}'

PY=""
for cand in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1 &&
     "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
    PY="$cand"; break
  fi
done
if [ -z "$PY" ]; then
  echo "ERROR: no Python >= 3.10 (stock macOS python3 is 3.9, too old for mlx)."
  echo "Fix: brew install python@3.12   (or: curl -LsSf https://astral.sh/uv/install.sh | sh)"
  exit 1
fi
echo "python: $($PY --version) ($(command -v $PY))"

WORK="${TMPDIR:-/tmp}/pr1456-validate"
rm -rf "$WORK" && mkdir -p "$WORK" && cd "$WORK"

echo "== fetch PR #1456 head =="
git init --quiet mlx-lm && cd mlx-lm
git remote add origin https://github.com/ml-explore/mlx-lm.git
git fetch --quiet --depth 1 origin pull/1456/head
git checkout --quiet FETCH_HEAD
echo "PR head: $(git rev-parse --short HEAD)"

echo "== venv + install =="
"$PY" -m venv .venv
source .venv/bin/activate
pip -q install --upgrade pip
# transformers 5.13 breaks mlx-lm import at this base (AutoTokenizer.register str key)
if ! pip install -e . "transformers<5.13" > pip_install.log 2>&1; then
  echo "pip install FAILED — last 15 lines:"; tail -15 pip_install.log; exit 1
fi
tail -1 pip_install.log
python -c "import mlx.core as mx; print('mlx device:', mx.default_device(), '| metal:', mx.metal.is_available())"

echo "== PR test files (GDN rollback + Rotating rollback + MTP) =="
python -m unittest tests.test_speculative_rollback tests.test_rotating_rollback tests.test_qwen3_5_mtp -v 2>&1 | tail -6
echo "== cache regression =="
python -m unittest tests.test_prompt_cache 2>&1 | tail -3

if [ "${1:-}" = "--e2e" ]; then
  echo "== e2e real weights: Qwen3.5-9B target + Qwen3.5-0.8B draft (both hybrid GDN) =="
  python - <<'EOF'
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

model, tok = load("mlx-community/Qwen3.5-9B-MLX-4bit")
draft, _ = load("mlx-community/Qwen3.5-0.8B-4bit")
try:
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
except Exception:
    pass
sampler = make_sampler(temp=0.0)

CODE = '''def load_config(path):
    with open(path) as f:
        return json.load(f)

def merge_configs(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = v
    return out
'''
prompts = [
    f"Rename the function `load_config` to `read_config` everywhere. Return the "
    f"COMPLETE file, in a python code block:\n```python\n{CODE}\n```",
    "Write a Python function that checks whether a string is a palindrome.",
]
ok = True
for ptext in prompts:
    try:
        ids = tok.apply_chat_template([{"role": "user", "content": ptext}],
                                      add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        ids = tok.apply_chat_template([{"role": "user", "content": ptext}],
                                      add_generation_prompt=True)
    runs = {}
    for name, kw in (("vanilla", {}), ("spec", dict(draft_model=draft, num_draft_tokens=4))):
        toks, last = [], None
        for r in stream_generate(model, tok, prompt=ids, max_tokens=250,
                                 sampler=sampler, **kw):
            toks.append(r.token); last = r
        runs[name] = toks
        print(f"  {name}: {last.generation_tokens} tok @ {last.generation_tps:.1f} tok/s")
        mx.clear_cache()
    same = runs["vanilla"] == runs["spec"]
    ok = ok and same
    print(f"  identical: {same}")
print("E2E", "PASS" if ok else "FAIL (token mismatch — report machine + output)")
EOF
fi
echo "== DONE =="
