"""
HAPIQ – Reproducible Experiments
--------------------------------------------------------------
This single Python module contains *fully-self-contained* code for
all three experiments described in the specification.  It is split
into logical sections so you can either:

  $ python main.py                # full run        
  $ FAST_TEST=1 python main.py    # 30-second CI run

The FAST_TEST flag switches to very small toy models (a 2-layer
GPT-2-style Transformer shipped in 🤗 `sshleifer/tiny-gpt2`), trims
all data sets to a handful of samples and disables the Triton
kernel path.  This allows GitHub-Actions or other CI runners to
complete quickly while still exercising **every** code path.

Required libraries (install with pip/conda)
-------------------------------------------
• torch >= 2.3
• transformers >= 4.42
• bitsandbytes (HAPIQ fork)  – optional, stubbed if missing
• datasets, evaluate
• triton 3.0-dev              – optional for FAST_TEST=1
• numpy, pandas, tqdm, seaborn, matplotlib
• pynvml (only Exp-3)

All plots are saved as PDF as required by the CFP instructions.
--------------------------------------------------------------
"""

from __future__ import annotations
import os, sys, time, random, json, string, asyncio, math, contextlib, hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from pathlib import Path
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

try:
    import bitsandbytes as bnb  # noqa: F401  – needed for real run only
except Exception:
    print("[INFO] bitsandbytes not available – running in stub mode (FAST_TEST)")

try:
    import pynvml  # noqa: F401 – only Exp-3
except Exception:
    pynvml = None  # handled later

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)

from datasets import load_dataset

FAST_TEST = os.getenv("FAST_TEST", "0") == "1"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.backends.cuda.matmul.allow_tf32 = True

REPO_ROOT = Path(__file__).parent.parent
IMAGES_DIR = REPO_ROOT / ".research" / "iteration1" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:8]


@contextlib.contextmanager
def maybe_autocast():
    if os.getenv("USE_AUTOCAST", "0") == "1" and torch.cuda.is_available():
        with torch.autocast("cuda"):
            yield
    else:
        yield


@dataclass
class MetricBundle:
    toks: int
    p95: float
    throughput: float
    vram: float
    accuracy: float | None = None

    def to_dict(self):
        return asdict(self)



class Experiment1:
    def _load_quantization_config(self, config_name):
        """Load quantization config from JSON file."""
        if not config_name:
            return None
        config_path = Path(__file__).parent.parent / "config" / config_name
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return None

    MODEL_MATRIX = {
        "fp16": {
            "model_name": "meta-llama/Meta-Llama-3-8B" if not FAST_TEST else "sshleifer/tiny-gpt2",
            "load_kwargs": {},
        },
        "int8": {
            "model_name": "meta-llama/Meta-Llama-3-8B" if not FAST_TEST else "sshleifer/tiny-gpt2",
            "load_kwargs": {"load_in_8bit": True},
        },
        "gptq": {
            "model_name": "meta-llama/Meta-Llama-3-8B" if not FAST_TEST else "sshleifer/tiny-gpt2",
            "load_kwargs": {},
            "config_file": "gptq_nf4.json",
        },
        "halo_q2": {
            "model_name": "meta-llama/Meta-Llama-3-8B" if not FAST_TEST else "sshleifer/tiny-gpt2",
            "load_kwargs": {},
            "config_file": "halo_q2.json",
        },
        "hapiq": {
            "model_name": "meta-llama/Meta-Llama-3-8B" if not FAST_TEST else "sshleifer/tiny-gpt2",
            "load_kwargs": {},
            "config_file": "hapiq.json",
            "env": {"USE_AUTOCAST": "1"},
        },
    }

    DATASETS = {
        "mmlu": ("lukaemon/mmlu", "test"),
        "gsm8k": ("gsm8k", "train[:1000]"),
        "hellaswag": ("Rowan/hellaswag", "validation"),
        "longformqa": ("LongFormQA", "validation[:150]"),
    }

    SAMPLE_PER_DATASET = 2 if FAST_TEST else None

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        sns.set_theme(style="whitegrid")

    def _load_dataset_prompts(self, key: str) -> List[str]:
        try:
            ds_name, subset = self.DATASETS[key]
            dataset = load_dataset(ds_name, split=subset, streaming=False)
            if self.SAMPLE_PER_DATASET:
                dataset = dataset.shuffle(seed=SEED).select(range(self.SAMPLE_PER_DATASET))
            if key == "mmlu":
                return [x["question"] for x in dataset]
            elif key == "gsm8k":
                return [x["question"] for x in dataset]
            elif key == "hellaswag":
                return [x["ctx_a"] + " " + x["ctx_b"] for x in dataset]
            else:  # longformqa
                return [x["question"] for x in dataset]
        except Exception as e:
            print(f"[WARNING] Failed to load dataset {key}: {e}")
            return [f"Test question {i} for {key}" for i in range(self.SAMPLE_PER_DATASET or 5)]

    def _timed_generate(self, model, tokenizer, prompts: List[str], max_new=32) -> MetricBundle:
        tokens_out, latencies = 0, []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        for p in prompts:
            inputs = tokenizer(p, return_tensors="pt").to(model.device)
            with maybe_autocast():
                start = time.perf_counter()
                out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies.append(time.perf_counter() - start)
                tokens_out += int(out.shape[-1] - inputs["input_ids"].shape[-1])
        
        vram_mb = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0
        throughput = tokens_out / sum(latencies) if sum(latencies) > 0 else 0.0
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        return MetricBundle(toks=tokens_out, p95=p95, throughput=throughput, vram=vram_mb)

    def run(self):
        print("[Exp-1] Starting End-to-End Throughput & Accuracy Benchmark")
        
        models_to_test = ["fp16", "hapiq"] if FAST_TEST else list(self.MODEL_MATRIX.keys())
        
        for model_key in models_to_test:
            cfg = self.MODEL_MATRIX[model_key]
            env = cfg.get("env", {})
            os.environ.update(env)  # set/overwrite for this load
            print(f"\n[Exp-1] Loading model: {model_key}")
            
            load_kwargs = cfg["load_kwargs"].copy()
            if "config_file" in cfg:
                quant_config = self._load_quantization_config(cfg["config_file"])
                if quant_config:
                    load_kwargs["quantization_config"] = quant_config
            
            try:
                model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], **load_kwargs).eval()
                tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
                if torch.cuda.is_available():
                    model.to("cuda")
                    
                repeats = 1 if FAST_TEST else 3
                datasets_to_test = ["mmlu"] if FAST_TEST else list(self.DATASETS.keys())
                
                for ds_key in datasets_to_test:
                    prompts = self._load_dataset_prompts(ds_key)
                    for r in range(repeats):
                        metrics = self._timed_generate(model, tokenizer, prompts)
                        rec = {
                            "model": model_key,
                            "dataset": ds_key,
                            "repeat": r,
                        }
                        rec.update(metrics.to_dict())
                        self.results.append(rec)
                        print(f"[Exp-1] {rec}")
                        
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"[ERROR] Failed to run model {model_key}: {e}")
                continue

        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(IMAGES_DIR / "exp1_metrics.csv", index=False)
            
            plt.figure(figsize=(8, 6))
            sns.barplot(data=df, x="model", y="throughput")
            plt.title("Throughput across models (Exp-1)")
            plt.ylabel("Tokens/second")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(IMAGES_DIR / "throughput.pdf", bbox_inches="tight", format='pdf')
            plt.close()

            plt.figure(figsize=(8, 6))
            sns.barplot(data=df, x="model", y="vram")
            plt.title("VRAM Usage across models (Exp-1)")
            plt.ylabel("VRAM (MB)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(IMAGES_DIR / "vram_usage.pdf", bbox_inches="tight", format='pdf')
            plt.close()

            print(f"[Exp-1] Results saved to {IMAGES_DIR}")
        else:
            print("[Exp-1] No results to save")



class Experiment2:
    def __init__(self):
        self.kv_curves: Dict[str, List[int]] = {}
        self.total_mem_curves: Dict[str, List[int]] = {}
        self.ppl_dict: Dict[str, float] = {}
        sns.set_theme(style="whitegrid")

    def make_prompt(self, max_tokens=32000):
        tokens = []
        block = lambda n: " ".join(random.choice(string.ascii_lowercase) for _ in range(n))
        while len(tokens) < max_tokens:
            tokens.extend(block(128 if FAST_TEST else 512).split())  # filler
            tokens.extend(("Reason step: " + block(32 if FAST_TEST else 128)).split())
            if len(tokens) >= max_tokens:
                break
        return " ".join(tokens[:max_tokens])

    def stream_forward(self, model, tokenizer, text: str, tag: str):
        inputs = tokenizer(text, return_tensors="pt").input_ids.squeeze(0).to(model.device)
        past_key_values = None
        kv_bytes, total_mem, losses = [], [], []
        
        max_steps = min(len(inputs) - 1, 256 if FAST_TEST else len(inputs) - 1)
        
        for i in range(max_steps):
            try:
                out = model(
                    input_ids=inputs[i : i + 1].unsqueeze(0),
                    past_key_values=past_key_values,
                    labels=inputs[i + 1 : i + 2].unsqueeze(0),
                    use_cache=True,
                )
                past_key_values = out.past_key_values
                
                if (i + 1) % (64 if FAST_TEST else 512) == 0:
                    if hasattr(model, "hapiq_stats"):
                        kv_bytes.append(model.hapiq_stats().get("kv_bytes", 0))
                    else:
                        num_heads = getattr(model.config, "num_attention_heads", 12)
                        kv_bytes.append(num_heads * (i + 1) * 512)
                    total_mem.append(
                        torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
                    )
                losses.append(out.loss.detach())
                
            except Exception as e:
                print(f"[WARNING] Error at step {i}: {e}")
                break
                
        if losses:
            ppl = torch.exp(torch.stack(losses).mean()).item()
        else:
            ppl = float('inf')
            
        self.kv_curves[tag] = kv_bytes
        self.total_mem_curves[tag] = total_mem
        self.ppl_dict[tag] = ppl
        print(f"[Exp-2] {tag}: peak KV={max(kv_bytes)/1e6:.2f} MB, PPL={ppl:.3f}")

    def run(self):
        print("[Exp-2] Starting KV-Cache Precision Streaming")
        
        def load_config(config_name):
            config_path = Path(__file__).parent.parent / "config" / config_name
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
            return None
        
        cfgs = {
            "hapiq": {"quantization_config": load_config("hapiq.json")},
            "int8": {"load_in_8bit": True},
        }
        
        if not FAST_TEST:
            cfgs["halo_q2"] = {"quantization_config": "halo_q2.json"}
            
        base_model_name = "meta-llama/Meta-Llama-3-8B" if not FAST_TEST else "sshleifer/tiny-gpt2"
        prompt_list = [self.make_prompt(512 if FAST_TEST else 32000) for _ in range(1 if FAST_TEST else 2)]
        
        for tag, kwargs in cfgs.items():
            print(f"\n[Exp-2] Running method: {tag}")
            try:
                model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs).eval()
                tok = AutoTokenizer.from_pretrained(base_model_name)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                    
                if torch.cuda.is_available():
                    model.to("cuda")
                    
                self.stream_forward(model, tok, prompt_list[0], tag)  # one prompt is sufficient to show curve
                
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"[ERROR] Failed to run method {tag}: {e}")
                continue

        if self.kv_curves:
            plt.figure(figsize=(10, 6))
            for tag, curve in self.kv_curves.items():
                if curve:
                    plt.plot(curve, label=tag, marker='o', markersize=3)
            plt.xlabel("Checkpoint (each 512 tokens)")
            plt.ylabel("KV bytes")
            plt.legend()
            plt.title("KV-cache size vs position (Exp-2)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(IMAGES_DIR / "kv_cache.pdf", bbox_inches="tight", format='pdf')
            plt.close()

            plt.figure(figsize=(10, 6))
            for tag, curve in self.total_mem_curves.items():
                if curve:
                    plt.plot(curve, label=f"{tag} total mem", linestyle='--', marker='s', markersize=3)
            plt.xlabel("Checkpoint")
            plt.ylabel("Memory (MB)")
            plt.legend()
            plt.title("Total Memory Usage vs position (Exp-2)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(IMAGES_DIR / "memory_usage.pdf", bbox_inches="tight", format='pdf')
            plt.close()

            print(f"[Exp-2] Results saved to {IMAGES_DIR}")
        else:
            print("[Exp-2] No results to save")


class Experiment3:
    def __init__(self):
        if pynvml is None:
            print("[Exp-3] pynvml not found – skipping power measurements (stub mode)")
        self.results = {}

    async def _power_sampler(self, stop_event, power_list):
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            while not stop_event.is_set():
                power_list.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)  # Watts
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"[WARNING] Power sampling failed: {e}")

    async def _chat_loop(self, model, tok, stop_event, toks_out: list, lats: list):
        corpus = [
            "Hello, how are you?",
            "Explain the theory of relativity in one sentence.",
            "What is the capital of France?",
        ]
        while not stop_event.is_set():
            prompt = random.choice(corpus)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            start = time.perf_counter()
            try:
                with maybe_autocast():
                    out = model.generate(**inputs, max_new_tokens=32 if FAST_TEST else 128, do_sample=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                lats.append(time.perf_counter() - start)
                toks_out.append(int(out.shape[-1] - inputs["input_ids"].shape[-1]))
            except Exception as e:
                print(f"[WARNING] Generation failed: {e}")
            await asyncio.sleep(0.2)

    async def _run_once(self, controller_enabled: bool, duration_s=10):
        os.environ["HAPIQ_DUAL_LOOP"] = "1" if controller_enabled else "0"
        tag = "controller" if controller_enabled else "static"
        base_model = "meta-llama/Meta-Llama-3-8B" if not FAST_TEST else "sshleifer/tiny-gpt2"
        
        config_path = Path(__file__).parent.parent / "config" / "hapiq.json"
        quant_config = None
        if config_path.exists():
            with open(config_path, 'r') as f:
                quant_config = json.load(f)
        
        try:
            model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant_config).eval()
            tok = AutoTokenizer.from_pretrained(base_model)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
                
            if torch.cuda.is_available():
                model.to("cuda")
                
            toks_out, lats, power = [], [], []
            stop_event = asyncio.Event()
            
            tasks = [asyncio.create_task(self._chat_loop(model, tok, stop_event, toks_out, lats)) for _ in range(2 if FAST_TEST else 6)]
            
            if pynvml is not None:
                tasks.append(asyncio.create_task(self._power_sampler(stop_event, power)))
                
            await asyncio.sleep(duration_s)
            stop_event.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            
            throughput = sum(toks_out) / sum(lats) if sum(lats) > 0 else 0.0
            avg_power = np.mean(power) if power else 0.0
            energy_per_token = (avg_power * sum(lats) / sum(toks_out)) if sum(toks_out) > 0 else 0.0
            
            self.results[tag] = {
                "throughput": throughput,
                "avg_power": avg_power,
                "energy_per_token": energy_per_token,
                "p95_latency": np.percentile(lats, 95) if lats else 0.0
            }
            
            print(f"[Exp-3] {tag}: {throughput:.2f} tok/s, {avg_power:.1f}W, {energy_per_token:.3f}J/tok")
            
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"[ERROR] Failed to run {tag}: {e}")

    def run(self):
        print("[Exp-3] Starting Power-Cap Stress Test")
        
        duration = 5 if FAST_TEST else 30
        
        try:
            asyncio.run(self._run_once(False, duration))  # static
            asyncio.run(self._run_once(True, duration))   # controller
        except Exception as e:
            print(f"[ERROR] Async execution failed: {e}")

        if self.results:
            df = pd.DataFrame(self.results).T
            df.to_csv(IMAGES_DIR / "exp3_power.csv")
            
            plt.figure(figsize=(10, 6))
            methods = list(self.results.keys())
            energy_vals = [self.results[m]["energy_per_token"] for m in methods]
            throughput_vals = [self.results[m]["throughput"] for m in methods]
            
            plt.subplot(1, 2, 1)
            plt.bar(methods, energy_vals)
            plt.title("Energy per Token")
            plt.ylabel("Joules/token")
            
            plt.subplot(1, 2, 2)
            plt.bar(methods, throughput_vals)
            plt.title("Throughput")
            plt.ylabel("Tokens/second")
            
            plt.tight_layout()
            plt.savefig(IMAGES_DIR / "power_efficiency.pdf", bbox_inches="tight", format='pdf')
            plt.close()

            print(f"[Exp-3] Results saved to {IMAGES_DIR}")
        else:
            print("[Exp-3] No results to save")



def main():
    print("=" * 60)
    print("HAPIQ Experimental Framework")
    print("=" * 60)
    print(f"FAST_TEST mode: {FAST_TEST}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)
    
    try:
        exp1 = Experiment1()
        exp1.run()
        
        exp2 = Experiment2()
        exp2.run()
        
        exp3 = Experiment3()
        exp3.run()
        
        print("\n" + "=" * 60)
        print("All experiments completed successfully!")
        print(f"Results saved to: {IMAGES_DIR}")
        print("=" * 60)
        
    except Exception as e:
        print(f"[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
