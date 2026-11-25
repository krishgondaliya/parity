**Target:** https://github.com/unslothai/unsloth-zoo/issues (the code is `unsloth_zoo/dataset_utils.py`; `unsloth.chat_templates.train_on_responses_only` re-exports it). Use the repo's "Bug / Issue" template: keep its six questions as they are (answered below), and replace its default "[Bug]" title with the one below.

**Title:** [train_on_responses_only] `SFTConfig.pad_to_multiple_of` is silently dropped with padded batches (unsloth-zoo 2026.11.12 and main)

**What happens**
With padded batches (`packing=False, padding_free=False`, as the recommended Qwen3 notebook sets) and `SFTConfig(pad_to_multiple_of=64)`, calling `train_on_responses_only(trainer, ...)` replaces the trainer's collator with `DataCollatorForSeq2Seq(tokenizer = tokenizer, **_kept)`. `_kept` is filled only when `_processor_backed` is true (a pad-delegating collator holding a processor); any other collator replaced here, TRL's `DataCollatorForLanguageModeling` included, gets a bare `DataCollatorForSeq2Seq(tokenizer = tokenizer)`. So batches are then padded only to their longest row. Nothing is printed about the swap. It reaches training: on a Tesla T4, the repro's longer variant plus one `trainer.train()` step drew a batch 25 wide, not 64.

Scope: only padded batches. With `padding_free` left unset and `max_steps=60` (as the Qwen3-4B-Thinking notebook sets), Unsloth chose `padding_free=True` on the T4; the collator was not replaced, and the batch was 64 wide. On the CPU route, Unsloth's defaults pick padded batches, so there it also applies with `padding_free` unset.

Labels are not affected, and on CPU the loss is the same up to float rounding. On the same two rows, the swapped collator's batch and the same collator class keeping `pad_to_multiple_of=64` have identical tokens and labels on every real position, and the same 18 supervised tokens. The loss was identical in 6 of 8 runs, and one float32 ulp (9.5e-7) apart in 2. Only the tensor width changes, which matters to anyone setting `pad_to_multiple_of` for fixed shapes (tensor-core alignment, compiled static shapes). Not measured on GPU.

The block: https://github.com/unslothai/unsloth-zoo/blob/cad81ef2dc557b1060896d3b294ef6b9a7f9c1ea/unsloth_zoo/dataset_utils.py#L2547-L2583. The comment at https://github.com/unslothai/unsloth-zoo/blob/cad81ef2dc557b1060896d3b294ef6b9a7f9c1ea/unsloth_zoo/dataset_utils.py#L2560-L2565 describes the same effect ("dropped its settings and silently turned the run into a dynamically padded one") for another collator class.

**Repro** (CPU, seconds once the tokenizer is cached; downloads Qwen3-0.6B's tokenizer and config only, the model is tiny and random)
```python
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
```

**Expected vs actual**
Expected: the batch padded to a multiple of 64, as it is before `train_on_responses_only` (64). Actual:
```
before train_on_responses_only: DataCollatorForLanguageModeling, batch width 64
after: DataCollatorForSeq2Seq(pad_to_multiple_of=None)
expected batch width: a multiple of 64; actual: 26
```
Exit code 1. Same output 3 of 3 times on 2026.11.10, on 2026.11.12, and on unsloth-zoo main at `cad81ef2`. (On a Tesla T4, a longer variant with the same calls gave 64, then 25.)

**Reference used**
`SFTConfig.pad_to_multiple_of`, as TRL's own collator applies it before `train_on_responses_only`.

**Template questions**
1. Did you update? `pip install --upgrade unsloth unsloth_zoo`
   Yes: also unsloth and unsloth-zoo 2026.11.12, and unsloth-zoo main at `cad81ef2`. Same result.
2. `Colab` or `Kaggle` or local / cloud
   Local (CPU), and Colab (Tesla T4).
3. Number GPUs used, use `nvidia-smi`
   None for the repro above; 1 Tesla T4 (driver 580.82.07, CUDA 13.0) for the GPU runs.
4. Which notebook? Please link!
   https://github.com/unslothai/notebooks/blob/ebe88117bb560d296910f6e733693294d630f024/nb/Qwen3_%2814B%29-Reasoning-Conversational.ipynb (its `SFTConfig` sets `padding_free = False`), with the `train_on_responses_only` call from https://github.com/unslothai/notebooks/blob/0c6268b8c23bb6e4377e0f8ccc4bbfce4876a3c4/nb/Qwen3_%284B%29-Thinking.ipynb
5. Which Unsloth version, TRL version, transformers version, PyTorch version?
   unsloth 2026.11.10, unsloth-zoo 2026.11.10, TRL 1.15.0, transformers 5.17.0, PyTorch 2.14.1, Python 3.12.13. Pinned requirements available on request.
6. Which trainer? `SFTTrainer`, `GRPOTrainer` etc
   `SFTTrainer`.

Happy to send a PR with a fix and a test if that would help.
