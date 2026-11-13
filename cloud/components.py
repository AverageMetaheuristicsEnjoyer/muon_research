import importlib
import json
import os
from pathlib import Path
import sys
import traceback
from urllib.request import urlopen

import torch
from transformers import T5ForConditionalGeneration

from Muon.muon import MuonWithAuxAdam
from tiger.data import build_seq2seq_dataset
from tiger.model import build_tiger_t5
from tiger.tokenizer import TigerSidTokenizer

category = sys.argv[1]
work = Path(os.environ["RECSYS_WORK"])
results = Path(os.environ["RECSYS_RESULTS"])
base = "https://raw.githubusercontent.com/zrc0622/MiniOneRec/6e4f769250b3538809fa5c251463f67ac44096ff/data/Amazon/"
summary = {"category": category, "checks": {}}
try:
    torch.manual_seed(42)
    torch.set_num_threads(2)
    index = work / "index.json"
    test = work / "test.csv"
    for local, remote in (
        (index, f"index/{category}.index.json"),
        (test, f"test/{category}_5_2016-10-2018-11.csv"),
    ):
        with urlopen(base + remote, timeout=120) as response:
            local.write_bytes(response.read())
    tokenizer = TigerSidTokenizer(str(index))
    dataset = build_seq2seq_dataset(str(test), tokenizer, sample=2)
    batch = {key: value for key, value in dataset[:2].items()}
    model = build_tiger_t5(tokenizer)
    muon, adam = [], []
    for name, parameter in model.named_parameters():
        (muon if parameter.ndim == 2 and name.endswith(".weight")
         and not any(token in name for token in ("shared", "embed", "lm_head", "relative_attention_bias"))
         else adam).append(parameter)
    optimizer = MuonWithAuxAdam([
        {"params": muon, "use_muon": True, "lr": 1e-4},
        {"params": adam, "use_muon": False, "lr": 1e-4},
    ])
    losses = []
    before = muon[0].detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        loss = model(**batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert (before - muon[0]).abs().max() > 0
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    checkpoint = work / "tiger-checkpoint"
    model.eval()
    model.save_pretrained(checkpoint)
    restored = T5ForConditionalGeneration.from_pretrained(checkpoint).eval()
    with torch.no_grad():
        assert torch.equal(model(**batch).logits, restored(**batch).logits)
        generated = restored.generate(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            max_new_tokens=tokenizer.n_digit + 1, num_beams=10, num_return_sequences=10,
            prefix_allowed_tokens_fn=tokenizer.build_prefix_allowed_tokens_fn(),
        )
    predictions = generated[:, 1:1 + tokenizer.n_digit].tolist()
    assert all(tuple(tokens) in tokenizer.valid_token_tuples for tokens in predictions)
    summary["checks"]["tiger_module_integration"] = {
        "losses": losses, "default_architecture": True, "checkpoint_reload_equal": True,
        "valid_catalog_predictions": len(predictions),
        "note": "Training loop supplied by test harness; no TIGER training entry point exists upstream.",
    }
    for module in ("experiments.generative.rl_muon", "tiger.evaluate"):
        try:
            importlib.import_module(module)
            summary["checks"][module] = "imported"
        except Exception:
            summary["checks"][module] = {"error": traceback.format_exc()}
    from trl import GRPOConfig
    config = GRPOConfig(output_dir=str(work / "rl-config"), use_cpu=True, bf16=False,
                        report_to="none", num_generations=2,
                        per_device_train_batch_size=2, per_device_eval_batch_size=2)
    summary["checks"]["rl_configuration_interface"] = {
        "max_prompt_length_present": hasattr(config, "max_prompt_length"),
        "required_at": "minionerec_engine/minionerec_trainer.py:350",
    }
    print("COMPONENT_SUMMARY", json.dumps(summary), flush=True)
finally:
    (results / "components.json").write_text(json.dumps(summary, indent=2))
