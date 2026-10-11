"""Generate the modern-LLM dataset: every prompt in modern_prompts.jsonl answered by every family below,
through the OpenAI-compatible NVIDIA endpoint (free tier, 40 RPM). Resumable: rows already in modern.jsonl are skipped.

  python data/generate_modern.py            # full run (~3 h)
  python data/generate_modern.py --limit 5  # dry run on 5 prompts
"""
import argparse, json, os, random, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://integrate.api.nvidia.com/v1/chat/completions"
KEY = os.environ["NVIDIA_API_KEY"]
# family -> (model id, extra request fields, max_tokens).  Reasoning models get a bigger budget so the
# answer is not swallowed by hidden reasoning; only message.content is kept.
# family -> (model id, extra request fields, max_tokens, workers). Visible answers are capped at ~450 tokens
# (the classifiers read at most 512 tokens anyway); reasoning models get a larger budget for hidden reasoning.
MODELS = {
    "Nemotron-3":     ("nvidia/nemotron-3-super-120b-a12b", {"chat_template_kwargs": {"enable_thinking": False}}, 450, 4),
    "GPT-OSS":        ("openai/gpt-oss-20b", {"reasoning_effort": "low"}, 1200, 8),
    "Kimi-K3":        ("moonshotai/kimi-k3", {}, 1200, 6),
    "Llama-3.2":      ("meta/llama-3.2-11b-vision-instruct", {}, 450, 10),
    "DiffusionGemma": ("google/diffusiongemma-26b-a4b-it", {}, 450, 4),
    # Tried and dropped (server-side queueing made them 0.25-1.2 responses/min; see report): google/gemma-4-31b-it,
    # z-ai/glm-5.3-flash, meta/llama-3.2-90b-vision-instruct. deepseek-ai/deepseek-v4.1-flash always timed out.
}
RPM = [38]            # per process; run one process per family with --rpm 6 to stay under the 40 RPM key limit
_lock, _last = threading.Lock(), [0.0]


def throttle():
    with _lock:
        wait = _last[0] + 60 / RPM[0] - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def call(model, extra, max_tokens, prompt, timeout=240):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": max_tokens, **extra}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for attempt in range(6):
        throttle(); t0 = time.time()
        try:
            r = json.load(urllib.request.urlopen(req, timeout=timeout))
            ch = r["choices"][0]
            content = (ch["message"].get("content") or "").strip()
            if content:
                return dict(response=content, finish_reason=ch.get("finish_reason"), latency=round(time.time() - t0, 1), usage=r.get("usage"))
            err = "empty content"
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}"
            if e.code in (400, 404):
                break
            if e.code == 429:                       # key-level rate limit: back off hard, do not add to the storm
                time.sleep(30 + 30 * attempt); continue
        except Exception as e:
            err = repr(e)
        time.sleep(5 * (attempt + 1))
    return dict(error=err)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int); ap.add_argument("--workers", type=int)
    ap.add_argument("--family", help="run only this family (one process per family)"); ap.add_argument("--rpm", type=int, default=38)
    args = ap.parse_args()
    RPM[0] = args.rpm
    fams = [args.family] if args.family else list(MODELS)
    prompts = [json.loads(l) for l in open(os.path.join(HERE, "modern_prompts.jsonl"), encoding="utf-8")]
    random.Random(0).shuffle(prompts)          # file is grouped by source; shuffled so any prefix is source-balanced
    prompts = prompts[: args.limit]
    suffix = f"_{args.family}" if args.family else ""
    out_path, err_path = os.path.join(HERE, f"modern{suffix}.jsonl"), os.path.join(HERE, f"modern_errors{suffix}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {(json.loads(l)["prompt_id"], json.loads(l)["family"]) for l in open(out_path, encoding="utf-8")}
    jobs = [(p, fam) for p in prompts for fam in fams if (p["prompt_id"], fam) not in done]
    workers = args.workers or (MODELS[fams[0]][3] if len(fams) == 1 else 12)
    print(f"{len(done)} done, {len(jobs)} to go", flush=True)
    out, errf, wlock = open(out_path, "a", encoding="utf-8"), open(err_path, "a", encoding="utf-8"), threading.Lock()
    n = [0, 0]

    def work(job):
        p, fam = job
        model, extra, mt, _ = MODELS[fam]
        res = call(model, extra, mt, p["instruction"])
        row = dict(prompt_id=p["prompt_id"], source=p["source"], instruction=p["instruction"], family=fam, model=model, ts=time.time(), **res)
        with wlock:
            (errf if "error" in res else out).write(json.dumps(row) + "\n"); (errf if "error" in res else out).flush()
            n["error" in res] += 1
            if sum(n) % 50 == 0:
                print(f"  {n[0]} ok / {n[1]} err / {len(jobs)} total", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, jobs))
    print(f"finished: {n[0]} ok, {n[1]} errors")


if __name__ == "__main__":
    main()
