# P-0005 impact: are labels or loss affected when train_on_responses_only drops pad_to_multiple_of?
# Same setup as repro.py (same pins). Compares the batch the swapped collator gives (setting dropped) with the same
# collator class keeping pad_to_multiple_of=64 (what the user asked for), on the same two rows, with the
# model in training mode as trainer.train() puts it: real tokens, labels, supervised-token count, and the model's loss.
import sys
import tempfile
import torch
import unsloth  # noqa: F401
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer

path, rev = tempfile.mkdtemp(), "c1899de289a04d12100db370d81485cdf75e47ca"
src = snapshot_download("Qwen/Qwen3-0.6B", revision=rev, allow_patterns=["*.json", "merges.txt"])
AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(src, num_hidden_layers=1, hidden_size=16, intermediate_size=32,
    num_attention_heads=2, num_key_value_heads=1, head_dim=8, layer_types=["full_attention"])).save_pretrained(path)
AutoTokenizer.from_pretrained(src).save_pretrained(path)
model, tokenizer = FastLanguageModel.from_pretrained(path, max_seq_length=2048, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=32, lora_alpha=32, target_modules=["q_proj", "v_proj"])
chats = [[{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello."}],
         [{"role": "user", "content": "Name the primary colors."}, {"role": "assistant", "content": "Red, yellow and blue."}]]
data = Dataset.from_dict({"text": [tokenizer.apply_chat_template(c, tokenize=False) for c in chats]})
args = SFTConfig(output_dir=path, dataset_text_field="text", pad_to_multiple_of=64, packing=False, padding_free=False)
trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=data, args=args)
trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
trainer.model.for_training()  # what trainer.train() does first; padding on the right
rows = [trainer.train_dataset[i] for i in range(2)]
dropped = trainer.data_collator(rows)
kept = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=64, label_pad_token_id=-100)(rows)


def real(batch, key):  # each row's values at its real (attention_mask == 1) positions
    return [[v for v, m in zip(r, a) if m] for r, a in zip(batch[key].tolist(), batch["attention_mask"].tolist())]


same_tokens, same_labels = real(dropped, "input_ids") == real(kept, "input_ids"), real(dropped, "labels") == real(kept, "labels")
pads_unsupervised = all(lab == -100 for b in (dropped, kept) for r, a in zip(b["labels"].tolist(), b["attention_mask"].tolist())
                        for lab, m in zip(r, a) if not m)
supervised = [int((b["labels"] != -100).sum()) for b in (dropped, kept)]
device = next(trainer.model.parameters()).device
with torch.no_grad():
    loss = [float(trainer.model(**{k: v.to(device) for k, v in b.items()}).loss) for b in (dropped, kept)]
print(f"batch width: dropped {dropped['input_ids'].shape[1]}, kept {kept['input_ids'].shape[1]}")
print(f"real tokens identical: {same_tokens}; labels on real tokens identical: {same_labels}; pads unsupervised: {pads_unsupervised}")
print(f"supervised tokens: dropped {supervised[0]}, kept {supervised[1]}")
print(f"loss: dropped {loss[0]!r}, kept {loss[1]!r}, |difference| {abs(loss[0] - loss[1]):.3g}")
sys.exit(0 if same_tokens and same_labels and pads_unsupervised and supervised[0] == supervised[1] else 1)
