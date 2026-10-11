"""modern.jsonl (raw API output) -> modern.csv.gz with the same columns as uf.csv.gz.
Keeps only prompts answered by EVERY family (complete block design), splits 80/10/10 grouped by prompt."""
import os, json, random, glob
import pandas as pd
from build import split_by_prompt, SOURCES

HERE = os.path.dirname(os.path.abspath(__file__))
FAMILIES = ["Nemotron-3", "GPT-OSS", "Kimi-K3", "Llama-3.2", "DiffusionGemma"]


def main():
    rows = {}
    files = [f for f in glob.glob(os.path.join(HERE, "modern*.jsonl")) if "errors" not in f and "prompts" not in f]
    for f in sorted(files):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["family"] in FAMILIES and len(r.get("response", "").strip()) >= 20:
                rows[(r["prompt_id"], r["family"])] = r      # last write wins -> dedupes retries
    df = pd.DataFrame(rows.values())
    fams = sorted(df.family.unique())
    complete = df.groupby("prompt_id").family.nunique()
    keep = complete[complete == len(fams)].index
    df = df[df.prompt_id.isin(keep)][["prompt_id", "source", "instruction", "model", "family", "response", "finish_reason", "latency"]]
    df = split_by_prompt(df)
    assert set(df[df.split == "train"].prompt_id).isdisjoint(df[df.split != "train"].prompt_id)
    df.to_csv(os.path.join(HERE, "modern.csv.gz"), index=False)
    print(f"{len(df)} rows, {len(keep)} prompts x {len(fams)} families; dropped {complete.lt(len(fams)).sum()} incomplete prompts")
    print(df.split.value_counts().to_dict())
    print(pd.crosstab(df.family, df.source))
    print("truncated (finish_reason=length) share by family:\n", df.groupby("family").finish_reason.apply(lambda s: (s == "length").mean()).round(2))
    print("median latency:\n", df.groupby("family").latency.median())


if __name__ == "__main__":
    main()
