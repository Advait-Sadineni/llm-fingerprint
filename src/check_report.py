"""Validation harness for the deliverable. `python -m src.check_report [--final]` exits non-zero on any failure.
Checks: no placeholders in the report, every referenced figure exists and is fresh, every headline neural cell has 3 seeds,
expected run counts per stage, the PDF compiles with the author on page 1, demo vocab matches the CNN checkpoint."""
import glob, json, os, re, subprocess, sys
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPST = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Typst.Typst_Microsoft.Winget.Source_8wekyb3d8bbwe\typst-x86_64-pc-windows-msvc\typst.exe")
fails = []


def check(cond, msg):
    (print("ok  ", msg) if cond else (fails.append(msg), print("FAIL", msg)))


def main(final=False):
    typ = "\n".join(open(f, encoding="utf-8").read() for f in glob.glob(os.path.join(HERE, "report", "**", "*.typ"), recursive=True))
    todos = re.findall(r"TODO_\w+|\w*PLACEHOLDER\w*|REPO_URL", typ)
    check(not todos or not final, f"placeholders in report: {sorted(set(todos))}")
    summ = os.path.join(HERE, "results", "summary.csv")
    for fig in sorted(set(re.findall(r'"/figures/([^"]+)"', typ))):
        p = os.path.join(HERE, "figures", fig)
        check(os.path.exists(p), f"figure referenced exists: {fig}")
        if os.path.exists(p):
            check(os.path.getmtime(p) >= os.path.getmtime(summ) - 1, f"figure newer than summary.csv: {fig}")
    s = pd.read_csv(summ, keep_default_na=False)
    head = s[(s.train_src == "") & (s.ablate == "none") & (s.frac == 1.0)]
    for data in ["uf"] + (["modern"] if "modern" in set(s.data) else []):
        for m in ["cnn", "lstm", "cnnlstm", "charcnn"]:
            for v in ["output", "input", "both"]:
                n = head[(head.data == data) & (head.model == m) & (head.view == v)].seeds
                check(len(n) and n.iloc[0] == 3, f"3 seeds: {data} {m} {v}")
        for v in ["output", "input", "both"]:
            n = head[(head.data == data) & (head.model == "tfidf") & (head.view == v)]
            check(len(n) == 1, f"tfidf present: {data} {v}")
        inp = head[(head.data == data) & (head.view == "input") & (head.model == "cnn")].acc_mean   # accuracy: macro-F1 can sit below chance when a model collapses to few classes
        chance = 1 / len(json.load(open(os.path.join(HERE, "results", f"{data}_cnn_output_s42.json")))["labels"])
        check(len(inp) and abs(inp.iloc[0] - chance) < 0.04, f"input-only accuracy at chance on {data}: {inp.iloc[0] if len(inp) else None:.3f} vs {chance:.3f}")
    check(len(s[(s.data == "uf") & (s.test_src != "")]) == 18, "LOSO uf: 18 cells")
    check(len(s[(s.data == "uf") & (s.ablate != "none")]) == 10, "ablations: 10 cells")
    check(len(s[(s.data == "uf") & (s.frac < 1)]) == 8, "learning curve: 8 cells")
    if final:
        check(len(s[(s.data == "modern") & (s.test_src != "")]) == 18, "LOSO modern: 18 cells")
        for f in ["paraphrase_uf.json", "crossgen.json", "features_modern.json", "explain_modern_cnn_output_s42.json"]:
            check(os.path.exists(os.path.join(HERE, "results", f)), f"results/{f} exists")
    # demo vocab matches checkpoint
    if os.path.exists(os.path.join(HERE, "demo", "vocab.json")):
        import torch
        ck = torch.load(os.path.join(HERE, "runs", "uf_cnn_output_s42.pt"), map_location="cpu")
        v = json.load(open(os.path.join(HERE, "demo", "vocab.json"), encoding="utf-8"))
        check(v["labels"] == ck["labels"] and len(v["stoi"]) == len(ck["itos"]), "demo vocab matches CNN checkpoint")
    # compile
    out = os.path.join(HERE, "report", "_check.pdf")
    r = subprocess.run([TYPST, "compile", "--root", HERE, os.path.join(HERE, "report", "report.typ"), out], capture_output=True, text=True)
    check(r.returncode == 0 and not r.stderr.strip(), f"typst compiles without warnings: {r.stderr.strip()[:200]}")
    if os.path.exists(out):
        from pypdf import PdfReader
        pdf = PdfReader(out); p1 = pdf.pages[0].extract_text()
        check("Advait Sadineni" in p1, "author on page 1")
        check(8 <= len(pdf.pages) <= 16, f"page count {len(pdf.pages)}")
        os.remove(out)
    print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES:\n- " + "\n- ".join(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main(final="--final" in sys.argv)
