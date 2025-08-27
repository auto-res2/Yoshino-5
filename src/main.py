import os
import sys
import json
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(__file__))

from preprocess import get_dataloaders
from train import TrainConfig, train, TinyModel
from evaluate import experiment1


def ensure_dirs():
    base = ".research/iteration1"
    images = f"{base}/images"
    os.makedirs(images, exist_ok=True)
    return base, images


def experiment3(images_dir: str):
    qset = [f"What is {i}+0?" for i in range(100)]
    configs = {
        "B1_graphbeam": dict(prune=True, graph_head=True),
        "B2_posthoc": dict(prune=False, graph_head=True),
        "B3_cot": dict(prune=False, graph_head=False),
    }
    results = {}
    for name, opts in configs.items():
        model = TinyModel(with_graph_head=opts["graph_head"])
        t0 = time.time()
        mem0 = psutil.Process().memory_info().rss / 2**20
        for q in qset:
            model.generate([q], return_graph=False, prune=opts["prune"])
        latency = (time.time() - t0) / len(qset)
        mempeak = psutil.Process().memory_info().rss / 2**20 - mem0
        results[name] = (latency, mempeak)
        print(f"{name:<14} p50-lat≈{latency*1000:6.1f} ms   ΔRAM={mempeak:5.1f} MB")
    x = list(results.keys())
    lat = [v[0] * 1000 for v in results.values()]
    plt.figure(figsize=(4, 3))
    sns.barplot(x=x, y=lat)
    plt.ylabel("Mean latency per query (ms)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    pdf = f"{images_dir}/inference_latency.pdf"
    plt.savefig(pdf, bbox_inches="tight")
    return {"latency_pdf": pdf}


def main():
    base, images = ensure_dirs()
    dls = get_dataloaders()
    print("== Train (toy) ==")
    tr_res = train(dls, images, TrainConfig(steps=60, lr=1e-2, with_ev=True, with_graph_head=True))
    print("== Eval (toy) ==")
    ev_res = experiment1(images)
    print("== Edge (toy) ==")
    ed_res = experiment3(images)
    status = {"status_enum": "stopped", "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(f"{base}/status.json", "w") as f:
        json.dump(status, f, indent=2)
    print(f'STATUS: {status["status_enum"]}')
    print("Artifacts:", tr_res, ev_res, ed_res)


if __name__ == "__main__":
    main()
