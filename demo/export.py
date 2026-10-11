"""Export a trained TextCNN to ONNX (+ vocab/labels JSON) for the in-browser demo.
  python demo/export.py runs/uf_cnn_output_s42.pt"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import build
from src import text as T

HERE = os.path.dirname(os.path.abspath(__file__))


def main(path):
    ck = torch.load(path, map_location="cpu")
    assert ck["model"] == "cnn"
    model = build("cnn", len(ck["itos"]), len(ck["labels"])); model.load_state_dict(ck["state"]); model.eval()
    x = torch.randint(3, 100, (1, 40))
    torch.onnx.export(model, x, os.path.join(HERE, "model.onnx"), input_names=["tokens"], output_names=["logits"],
                      dynamic_axes={"tokens": {0: "batch", 1: "length"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    json.dump(dict(stoi={t: i for i, t in enumerate(ck["itos"])}, labels=ck["labels"], max_len=T.MAX_OUT, pattern=T.TOKEN_RE.pattern),
              open(os.path.join(HERE, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
    # parity check: ONNX output must match PyTorch on a real sentence
    import onnxruntime as ort
    toks = T.tokenize("Sure! Here are three tips:\n- Practice daily.\n- Taste as you go.\n- Keep knives sharp.")
    ids = [ck["itos"].index(t) if t in ck["itos"] else 1 for t in toks]
    ids += [0] * max(0, 5 - len(ids))
    ref = model(torch.tensor([ids])).detach().numpy()
    out = ort.InferenceSession(os.path.join(HERE, "model.onnx")).run(None, {"tokens": np.array([ids], dtype=np.int64)})[0]
    assert np.allclose(ref, out, atol=1e-3), (ref, out)
    print("exported", os.path.getsize(os.path.join(HERE, "model.onnx")) / 1e6, "MB; parity ok; labels", ck["labels"])


if __name__ == "__main__":
    main(sys.argv[1])
