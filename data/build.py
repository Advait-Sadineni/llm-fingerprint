"""Build the classification datasets from raw UltraFeedback jsonl.

Outputs (all csv.gz, columns: prompt_id, source, instruction, model, family, response, split):
  uf.csv.gz          5 families x 4,000, IDENTICAL source mix per family (controlled)
  uf_natural.csv.gz  5 families x 4,000, each family's natural source mix (uncontrolled; RQ2 probe)
  uf17.csv.gz        16 fine-grained model labels x 1,500 (pythia-12b dropped: only 834 rows)
  modern_prompts.jsonl  1,200 prompts (from uf TEST prompts) for the modern-model generation
Splits are grouped by prompt_id (80/10/10) so no prompt appears in two splits.
"""
import os, json, random, collections
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ["evol_instruct", "false_qa", "flan", "sharegpt", "truthful_qa", "ultrachat"]
FAMILY = {"gpt-4": "GPT", "gpt-3.5-turbo": "GPT",
          "llama-2-7b-chat": "Llama-2", "llama-2-13b-chat": "Llama-2", "llama-2-70b-chat": "Llama-2",
          "bard": "Bard", "falcon-40b-instruct": "Falcon", "mpt-30b-chat": "MPT"}
FAMILIES = ["GPT", "Llama-2", "Bard", "Falcon", "MPT"]
# per-source quota per family (sums to 4,000); bounded by Bard's availability (flan 701, truthful_qa 204)
QUOTA = {"evol_instruct": 1000, "false_qa": 450, "flan": 600, "sharegpt": 1000, "truthful_qa": 150, "ultrachat": 800}
PER_FAMILY = sum(QUOTA.values())
FINE_PER_MODEL = 1500
MODERN_QUOTA = {"evol_instruct": 300, "false_qa": 135, "flan": 180, "sharegpt": 300, "truthful_qa": 45, "ultrachat": 240}
SEED = 42


def load():
    rows, seen = [], set()
    for src in SOURCES:
        for i, line in enumerate(open(os.path.join(HERE, "raw", src + ".jsonl"), encoding="utf-8")):
            r = json.loads(line)
            for c in r["completions"]:
                resp = (c["response"] or "").strip()
                if len(resp) < 20 or resp in seen:
                    continue
                seen.add(resp)
                rows.append(dict(prompt_id=f"{src}:{i}", source=src, instruction=r["instruction"].strip(),
                                 model=c["model"], response=resp))
    return pd.DataFrame(rows)


def split_by_prompt(df, seed=SEED):
    ids = sorted(df.prompt_id.unique())
    random.Random(seed).shuffle(ids)
    n = len(ids)
    split = {p: "train" for p in ids[: int(0.8 * n)]}
    split.update({p: "val" for p in ids[int(0.8 * n): int(0.9 * n)]})
    split.update({p: "test" for p in ids[int(0.9 * n):]})
    return df.assign(split=df.prompt_id.map(split))


def sample_controlled(df):
    parts = []
    for fam in FAMILIES:
        for src, q in QUOTA.items():
            pool = df[(df.family == fam) & (df.source == src)]
            assert len(pool) >= q, (fam, src, len(pool))
            parts.append(pool.sample(q, random_state=SEED))
    return pd.concat(parts)


def sample_natural(df):
    return pd.concat(df[df.family == f].sample(PER_FAMILY, random_state=SEED) for f in FAMILIES)


def check(df, label_col, per_label):
    for s in ["train", "val", "test"]:
        assert set(df[df.split == s].source) == set(SOURCES), s
    tr, te = set(df[df.split == "train"].prompt_id), set(df[df.split != "train"].prompt_id)
    assert not (tr & te), "prompt leakage"
    assert (df[label_col].value_counts() == per_label).all(), df[label_col].value_counts()


def main():
    raw = load()
    print("raw completions after filtering:", len(raw))
    fam = raw[raw.model.isin(FAMILY)].assign(family=lambda d: d.model.map(FAMILY))

    uf = split_by_prompt(sample_controlled(fam))
    check(uf, "family", PER_FAMILY)
    uf.to_csv(os.path.join(HERE, "uf.csv.gz"), index=False)

    nat = split_by_prompt(sample_natural(fam))
    check(nat, "family", PER_FAMILY)
    nat.to_csv(os.path.join(HERE, "uf_natural.csv.gz"), index=False)

    fine = raw[raw.model != "pythia-12b"].assign(family=lambda d: d.model)
    fine = split_by_prompt(pd.concat(fine[fine.model == m].sample(FINE_PER_MODEL, random_state=SEED) for m in sorted(fine.model.unique())))
    check(fine, "model", FINE_PER_MODEL)
    fine.to_csv(os.path.join(HERE, "uf17.csv.gz"), index=False)

    test_prompts = uf[uf.split == "test"].drop_duplicates("prompt_id")
    test_prompts = test_prompts[test_prompts.instruction.str.len().between(10, 2000)]
    with open(os.path.join(HERE, "modern_prompts.jsonl"), "w", encoding="utf-8") as f:
        for src, q in MODERN_QUOTA.items():
            pool = test_prompts[test_prompts.source == src]
            assert len(pool) >= q, (src, len(pool))
            for _, r in pool.sample(q, random_state=SEED).iterrows():
                f.write(json.dumps(dict(prompt_id=r.prompt_id, source=src, instruction=r.instruction)) + "\n")

    for name, d, col in [("uf", uf, "family"), ("uf_natural", nat, "family"), ("uf17", fine, "model")]:
        print(f"\n{name}: {len(d)} rows; split sizes {d.split.value_counts().to_dict()}")
        print(pd.crosstab(d[col], d.source))
    print("\nbuild.py ok")


if __name__ == "__main__":
    main()
