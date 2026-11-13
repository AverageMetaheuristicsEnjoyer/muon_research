import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from urllib.request import urlopen

import pandas as pd
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Qwen2Config, Qwen2ForCausalLM

from Muon.muon import MuonWithAuxAdam
from minionerec_engine.data import (
    FusionSeqRecDataset, SFTData, SidItemFeatDataset,
    SidSFTDataset, TitleHistory2SidSFTDataset,
)

mode, category = sys.argv[1:]
work = Path(os.environ["RECSYS_WORK"])
results = Path(os.environ["RECSYS_RESULTS"])
data = work / "data"
revision = "6e4f769250b3538809fa5c251463f67ac44096ff"
source = f"https://raw.githubusercontent.com/zrc0622/MiniOneRec/{revision}/data/Amazon/"
stem = f"{category}_5_2016-10-2018-11"
summary = {"mode": mode, "category": category, "data_revision": revision, "checks": {}}


def save():
    (results / "summary.json").write_text(json.dumps(summary, indent=2))


def run(*args):
    print("COMMAND", list(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


try:
    torch.set_num_threads(2)
    torch.manual_seed(42)
    summary["code_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    summary["versions"] = {name: importlib.metadata.version(name) for name in (
        "torch", "transformers", "datasets", "accelerate", "pandas", "fire",
    )}
    summary["cuda_available"] = torch.cuda.is_available()
    if mode == "gpu":
        assert torch.cuda.is_available(), "GPU rehearsal received no CUDA device"
        summary["gpu"] = torch.cuda.get_device_name(0)
        summary["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
    print("ENVIRONMENT", json.dumps(summary), flush=True)

    paths = [f"{split}/{stem}.csv" for split in ("train", "valid", "test")]
    paths += [f"index/{category}.{suffix}.json" for suffix in ("index", "item")]
    paths += [f"info/{stem}.txt"]
    hashes = {}
    for name in paths:
        destination = data / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(source + name, timeout=120) as response:
            content = response.read()
        destination.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    summary["data_sha256"] = hashes
    index_path = data / f"index/{category}.index.json"
    item_path = data / f"index/{category}.item.json"
    index = json.loads(index_path.read_text())
    catalog = {"".join(sid) for sid in index.values()}
    counts = {}
    for split in ("train", "valid", "test"):
        frame = pd.read_csv(data / split / f"{stem}.csv")
        assert {"history_item_sid", "item_sid", "item_id"} <= set(frame.columns)
        assert set(frame.item_sid) <= catalog, f"Unknown target SID in {split}"
        counts[split] = len(frame)
    summary["checks"]["data"] = {"rows": counts, "items": len(index)}
    original_info = Path(f"minionerec_engine/data/Amazon/info/{stem}.txt")
    summary["info_matches_muonrec"] = original_info.read_bytes() == (data / f"info/{stem}.txt").read_bytes()
    save()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.add_tokens(sorted({token for sid in index.values() for token in sid}))
    train_file = data / "train" / f"{stem}.csv"
    common = dict(tokenizer=tokenizer, max_len=128, sample=2, seed=42, category=category)
    datasets = [
        SidSFTDataset(train_file=str(train_file), **common),
        SidItemFeatDataset(item_file=str(item_path), index_file=str(index_path), **common),
        FusionSeqRecDataset(train_file=str(train_file), item_file=str(item_path), index_file=str(index_path), **common),
        SFTData(train_file=str(train_file), **common),
        TitleHistory2SidSFTDataset(train_file=str(train_file), item_file=str(item_path), index_file=str(index_path), **common),
    ]
    summary["checks"]["sft_datasets"] = {}
    for dataset in datasets:
        assert len(dataset) > 0
        assert all(any(label != -100 for label in dataset[i]["labels"]) for i in range(len(dataset)))
        summary["checks"]["sft_datasets"][type(dataset).__name__] = len(dataset)

    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)
    model = Qwen2ForCausalLM(config)
    muon, adam = [], []
    for name, parameter in model.named_parameters():
        (muon if parameter.ndim == 2 and "proj" in name else adam).append(parameter)
    optimizer = MuonWithAuxAdam([
        {"params": muon, "use_muon": True, "lr": 1e-4},
        {"params": adam, "use_muon": False, "lr": 1e-4},
    ])
    batch = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")([datasets[0][0]])
    before = muon[0].detach().clone()
    loss = model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    delta = (muon[0] - before).abs().max().item()
    assert delta > 0, "Muon projection did not change"
    summary["checks"]["cpu_tiny_qwen_muon_step"] = {"loss": loss.item(), "max_projection_change": delta}
    for module in ("experiments.generative.rl_muon", "tiger.evaluate"):
        try:
            importlib.import_module(module)
            summary["checks"][module] = "imported"
        except Exception as error:
            summary["checks"][module] = {"import_error": f"{type(error).__name__}: {error}"}
    del model, optimizer, datasets
    save()

    if mode == "gpu":
        output = work / "sft_muon"
        run(sys.executable, "experiments/generative/sft_muon.py",
            "--base_model=Qwen/Qwen2.5-1.5B-Instruct",
            f"--train_file={train_file}", f"--eval_file={data / 'valid' / (stem + '.csv')}",
            f"--output_dir={output}", "--sample=2", "--batch_size=2", "--micro_batch_size=1",
            "--num_epochs=1", "--cutoff_len=128", f"--category={category}",
            f"--sid_index_path={index_path}", f"--item_meta_path={item_path}", "--muon_lr=0.0001")
        final = output / "final_checkpoint"
        assert (final / "config.json").exists() and (final / "tokenizer_config.json").exists()
        states = list(output.glob("checkpoint-*/trainer_state.json"))
        summary["checks"]["sft"] = json.loads(states[-1].read_text())
        (results / "trainer_state.json").write_text(json.dumps(summary["checks"]["sft"], indent=2))
        small_test = work / "test.csv"
        pd.read_csv(data / "test" / f"{stem}.csv").head(2).to_csv(small_test, index=False)
        predictions_path = results / "predictions.json"
        run(sys.executable, "minionerec_engine/evaluate.py", f"--base_model={final}",
            f"--info_file={data / 'info' / (stem + '.txt')}", f"--category={category}",
            f"--test_data_path={small_test}", f"--result_json_data={predictions_path}",
            "--batch_size=1", "--num_beams=10", "--max_new_tokens=8")
        predictions = json.loads(predictions_path.read_text())
        assert len(predictions) == 2
        metrics = {}
        for k in (1, 3, 5, 10):
            hits, discounts = [], []
            for record in predictions:
                candidates = [str(value).strip() for value in record["predict"]]
                assert len(candidates) == 10
                assert all(value in catalog for value in candidates), f"Non-catalog prediction: {candidates}"
                target = str(record["output"]).strip()
                rank = candidates.index(target) if target in candidates else None
                hit = rank is not None and rank < k
                hits.append(float(hit))
                discounts.append(1 / math.log2(rank + 2) if hit else 0)
            metrics[f"Recall@{k}"] = sum(hits) / len(hits)
            metrics[f"NDCG@{k}"] = sum(discounts) / len(discounts)
        summary["checks"]["evaluation"] = metrics
        run(sys.executable, "minionerec_engine/calc.py", f"--path={predictions_path}",
            f"--item_path={data / 'info' / (stem + '.txt')}")
    summary["status"] = "cpu_core_passed" if mode == "cpu" else "gpu_sft_evaluation_passed"
    print("RECSYS_SUMMARY", json.dumps(summary), flush=True)
except Exception:
    summary["status"] = "failed"
    summary["error"] = traceback.format_exc()
    raise
finally:
    save()
