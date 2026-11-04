"""Download fresh Corpus-200B shards, disjoint from the training pool, for downstream validation.

Downloads documents, DCLM and FineWeb-Edu scores, topic/format/cluster labels and token counts;
the remaining features are extracted locally with pipeline/features/. The paper uses 30 shards.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import M2_DATA, CORPUS_DIR  # noqa: E402
from config import DATA_DIR, REPO_ROOT  # noqa

FRESH_SEED = 4242
TOTAL_SHARDS = 9888


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    sampled = json.loads((M2_DATA / "sampled_shards.json").read_text())
    excluded = set(sampled["shards"])
    rng = random.Random(FRESH_SEED)
    candidates = [i for i in range(TOTAL_SHARDS) if i not in excluded]
    fresh = sorted(rng.sample(candidates, args.n))

    out = {"seed": FRESH_SEED, "shards": fresh}
    (DATA_DIR / "fresh_shards.json").write_text(json.dumps(out, indent=2))
    print(f"Picked {args.n} fresh shards: {fresh}")
    if not args.download:
        print("Dry run. Pass --download to fetch.")
        return

    from huggingface_hub import hf_hub_download
    from tqdm import tqdm
    REPO = "WebOrganizer/Corpus-200B"
    out_root = CORPUS_DIR
    items = [
        ("documents", ".jsonl.zst"),
        ("scores_dclm-fasttext", ".npy"),
        ("scores_fineweb-edu__rounded", "__rounded.npy"),
        ("domains_topics", "__choice.npy"),
        ("domains_formats", "__choice.npy"),
        ("domains_clusters-k24", ".npy"),
        ("tokens", ".npy"),
    ]
    for s in tqdm(fresh, desc="shards"):
        stem = f"CC_shard_{s:08d}_processed"
        for sub, suf in items:
            fname = f"{sub}/{stem}{suf}"
            try:
                hf_hub_download(repo_id=REPO, filename=fname,
                                 repo_type="dataset",
                                 local_dir=str(out_root))
            except Exception as e:
                tqdm.write(f"  shard {s} {sub}: {e}")


if __name__ == "__main__":
    main()
