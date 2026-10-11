"""RQ4 paraphrase attack: rewrite 300 UltraFeedback TEST responses (60 per family) with one modern model, keeping content.
Writes data/paraphrased_uf.jsonl. Then `python -m src.paraphrase_eval` scores every saved classifier before/after."""
import json, os, random, threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import generate_modern
from generate_modern import call
generate_modern.RPM[0] = 10         # share the 40 RPM key with the running generators

HERE = os.path.dirname(os.path.abspath(__file__))
PARAPHRASER = ("meta/llama-3.2-11b-vision-instruct", {}, 900)     # Nemotron-3 was first choice but its endpoint returned 429 all evening
PROMPT = ("Rewrite the following text in your own words. Keep every fact, instruction and the overall meaning, but change the wording, "
          "sentence structure and formatting freely. Output only the rewritten text.\n\n<text>\n{t}\n</text>")


def main(n_per_family=60):
    df = pd.read_csv(os.path.join(HERE, "uf.csv.gz"), keep_default_na=False)
    df = df[(df.split == "test") & (df.response.str.len() < 2500)]
    rows = pd.concat(df[df.family == f].sample(n_per_family, random_state=0) for f in sorted(df.family.unique()))
    out_path = os.path.join(HERE, "paraphrased_uf.jsonl")
    done = {json.loads(l)["prompt_id"] + json.loads(l)["family"] for l in open(out_path, encoding="utf-8")} if os.path.exists(out_path) else set()
    todo = [r for r in rows.itertuples() if r.prompt_id + r.family not in done]
    print(len(done), "done,", len(todo), "to go", flush=True)
    lock, n = threading.Lock(), [0]

    def work(r):
        res = call(*PARAPHRASER, PROMPT.format(t=r.response))
        with lock:
            if "error" in res:
                print("ERR", r.prompt_id, res["error"], flush=True); return
            f.write(json.dumps(dict(prompt_id=r.prompt_id, source=r.source, instruction=r.instruction, family=r.family, paraphraser=PARAPHRASER[0],
                                    response=r.response, paraphrase=res["response"], latency=res["latency"])) + "\n"); f.flush()
            n[0] += 1
            if n[0] % 25 == 0:
                print(n[0], "/", len(todo), flush=True)

    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(4) as ex:
        list(ex.map(work, todo))
    print("done")


if __name__ == "__main__":
    main()
