"""Download the six UltraFeedback jsonl files (openbmb/UltraFeedback, MIT) into data/raw/.
Plain urllib; resumable by skipping files whose size already matches."""
import os, sys, urllib.request
BASE = "https://huggingface.co/datasets/openbmb/UltraFeedback/resolve/main/"
FILES = ["evol_instruct", "false_qa", "flan", "sharegpt", "truthful_qa", "ultrachat"]
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
os.makedirs(RAW, exist_ok=True)
for name in FILES:
    url, dst = BASE + name + ".jsonl", os.path.join(RAW, name + ".jsonl")
    size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        print(f"skip {name} ({size/1e6:.0f} MB)"); continue
    print(f"get  {name} ({size/1e6:.0f} MB)", flush=True)
    urllib.request.urlretrieve(url, dst + ".part")
    os.replace(dst + ".part", dst)
print("done")
