# unsloth==2026.11.10, unsloth-zoo==2026.11.10, trl==1.15.0, transformers==5.17.0, torch==2.14.1, Python 3.12.
# train_on_responses_only drops SFTConfig.pad_to_multiple_of when batches are padded (packing=False, padding_free=False,
# as the recommended Qwen3 notebook sets). Downloads Qwen3-0.6B's tokenizer and config only; the model is tiny and random.
import sys, tempfile
import unsloth  # noqa: F401  (first, as the notebooks do)
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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
width = lambda t: t.data_collator([t.train_dataset[i] for i in range(2)])["input_ids"].shape[1]
print(f"before train_on_responses_only: {type(trainer.data_collator).__name__}, batch width {width(trainer)}")
trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
print(f"after: {type(trainer.data_collator).__name__}(pad_to_multiple_of={trainer.data_collator.pad_to_multiple_of})")
print(f"expected batch width: a multiple of 64; actual: {width(trainer)}")
sys.exit(0 if width(trainer) % 64 == 0 else 1)
