"""P-0004 triage repro (not shrunk): Unsloth 2026.11.10 on CPU appends a second <|im_end|> to chat-formatted text rows;
its GPU-route dataset preparation (unsloth_zoo's sft_prepare_dataset) leaves the same text alone. CPU only; downloads
Qwen3-0.6B's tokenizer and config files, builds a tiny random-weight model. Exit 1 if the two routes differ."""
import os, shutil, sys, tempfile
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
import unsloth  # noqa: F401  (import first, as the notebooks do)
from unsloth import FastLanguageModel
import datasets, torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from unsloth_zoo.dataset_utils import sft_prepare_dataset

src = snapshot_download("Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca",
                        allow_patterns=["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json"])
folder = os.path.join(tempfile.mkdtemp(), "tiny-qwen3")
config = AutoConfig.from_pretrained(src)
for k, v in dict(num_hidden_layers=1, hidden_size=16, intermediate_size=32, num_attention_heads=2, num_key_value_heads=1, head_dim=8).items():
    setattr(config, k, v)
config.layer_types = config.layer_types[:1] if getattr(config, "layer_types", None) else None
torch.manual_seed(0); AutoModelForCausalLM.from_config(config).save_pretrained(folder)
for f in os.listdir(src):
    if f != "config.json": shutil.copy(os.path.join(src, f), folder)

model, tokenizer = FastLanguageModel.from_pretrained(model_name=folder, max_seq_length=2048, load_in_4bit=True)
messages = [{"role": "user", "content": "Name the primary colors."}, {"role": "assistant", "content": "Red, yellow and blue."}]
text = tokenizer.apply_chat_template(messages, tokenize=False)  # as Unsloth's Qwen3 notebook formats data
trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=datasets.Dataset.from_dict({"text": [text]}),
                     args=SFTConfig(output_dir=tempfile.mkdtemp(), dataset_text_field="text", report_to="none"))
cpu = list(trainer.train_dataset[0]["input_ids"])
gpu = list(sft_prepare_dataset(trainer, datasets.Dataset.from_dict({"text": [text]}), tokenizer, trainer.args, False, None, "train")[0]["input_ids"])
render = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True)["input_ids"]
print("chat template render ends:", repr(tokenizer.decode(render[-3:])))
print("CPU route (what this run trains on) ends:", repr(tokenizer.decode(cpu[-3:])), "| equals render:", cpu == render)
print("GPU route preparation, same text, ends: ", repr(tokenizer.decode(gpu[-3:])), "| equals render:", gpu == render)
sys.exit(1 if cpu != gpu else 0)
