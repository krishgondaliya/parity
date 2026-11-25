"""P-0005 triage repro (not shrunk): Unsloth's train_on_responses_only drops SFTConfig.pad_to_multiple_of when it swaps in
its own collator, without a message. CPU; downloads Qwen3-0.6B's tokenizer and config files, builds a tiny random-weight
model. Exit 1 if the setting is dropped. Padded batches are set explicitly (packing=False, padding_free=False): on GPU,
Unsloth may otherwise choose packing or padding-free, and then the collator is never replaced."""
import os, shutil, sys, tempfile
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
import unsloth  # noqa: F401  (import first, as the notebooks do)
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
import datasets, torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

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
chats = [[{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello."}],
         [{"role": "user", "content": "Name the primary colors."}, {"role": "assistant", "content": "Red, yellow and blue."}]]
texts = [tokenizer.apply_chat_template(c, tokenize=False) for c in chats]
trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=datasets.Dataset.from_dict({"text": texts}),
                     args=SFTConfig(output_dir=tempfile.mkdtemp(), dataset_text_field="text", report_to="none", pad_to_multiple_of=64,
                                   packing=False, padding_free=False))  # padded batches, as the recommended Qwen3 notebook sets
rows = [trainer.train_dataset[i] for i in range(2)]
before = (type(trainer.data_collator).__name__, getattr(trainer.data_collator, "pad_to_multiple_of", None), trainer.data_collator(rows)["input_ids"].shape[1])
trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
rows = [trainer.train_dataset[i] for i in range(2)]
after = (type(trainer.data_collator).__name__, getattr(trainer.data_collator, "pad_to_multiple_of", None), trainer.data_collator(rows)["input_ids"].shape[1])
print("SFTConfig(pad_to_multiple_of=64)")
print("before train_on_responses_only: collator %s(pad_to_multiple_of=%s), batch width %d" % before)
print("after  train_on_responses_only: collator %s(pad_to_multiple_of=%s), batch width %d" % after)
sys.exit(1 if after[1] != 64 or after[2] % 64 else 0)
