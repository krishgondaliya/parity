# P-0005 on Unsloth's GPU route

Run `parity-gpu-results-25c3dc83997d-20261125T185944Z.zip`: commit `25c3dc83997d`, Tesla T4 (driver 580.82.07), torch 2.14.1+cu130, Unsloth on `cuda`, `envs/unsloth-2026.11.10/uv.lock`. Verified: 12 files against their manifests, `git archive` of the commit rebuilt here with the recorded sha256, every step passed, every installed version as locked (`provenance.json`).

## padded (exit 1)

The repro with packing=False, padding_free=False, as the recommended Qwen3 notebook sets. Its rule: exit 1 if pad_to_multiple_of was dropped, 0 if kept.

```
SFTConfig(pad_to_multiple_of=64)
batching: packing=False, padding_free=False (set explicitly)
before train_on_responses_only: collator DataCollatorForLanguageModeling(pad_to_multiple_of=64), batch width 64
after  train_on_responses_only: collator DataCollatorForSeq2Seq(pad_to_multiple_of=None), batch width 25
padding: the shorter row is padded on the left; collator tokenizer padding_side='left'
```

## padding_free_unset (exit 0)

The repro with packing and padding_free left to Unsloth's defaults, max_steps=60 (the Qwen3-4B-Thinking-2507 notebook). The 'batching:' line shows what Unsloth chose; under packing or padding-free the batch is one flattened row, so read the printed lines, not the exit code.

```
SFTConfig(pad_to_multiple_of=64)
batching: packing=False, padding_free=True (left to Unsloth's defaults, max_steps=60)
before train_on_responses_only: collator DataCollatorForLanguageModeling(pad_to_multiple_of=64), batch width 64
after  train_on_responses_only: collator DataCollatorForLanguageModeling(pad_to_multiple_of=64), batch width 64
padding: the shorter row is padded on the none; collator tokenizer padding_side=None
```

## padded_train_step (exit 1)

The padded repro plus one real trainer.train() step, with the collator wrapped to record the batch training actually draws (width and padding side). Unsloth's train() calls model.for_training() first, so this is the training state; the 'padding:' line before it is the state SFTTrainer.__init__ leaves.

```
SFTConfig(pad_to_multiple_of=64)
batching: packing=False, padding_free=False (set explicitly)
before train_on_responses_only: collator DataCollatorForLanguageModeling(pad_to_multiple_of=64), batch width 64
after  train_on_responses_only: collator DataCollatorForSeq2Seq(pad_to_multiple_of=None), batch width 25
padding: the shorter row is padded on the left; collator tokenizer padding_side='left'
train step: batch width 25 during trainer.train(); the shorter row is padded on the right
```

## adapter_check (exit 0)

Rule 4: Parity's own adapter at the pad_to t1.special_tokens asks, with objective all (no train_on_responses_only) and assistant_only. Exit 0 if the mapping reaches the trainer (objective all pads to pad_to), 1 if it doesn't (the adapter is the suspect).

```
backend gpu; unpadded rows [25, 537]; pad_to=545 (the longer row + 8, as t1.special_tokens asks)
objective all: batch width 545; pad_token_id: DataCollatorForLanguageModeling.tokenizer.pad_token_id=151643, pad_to_multiple_of=545 (trainer.args.pad_to_multiple_of=545), label_pad_token_id=None, padding_side=None
objective assistant_only: batch width 537; pad_token_id: DataCollatorForSeq2Seq.tokenizer.pad_token_id=151643, pad_to_multiple_of=None (trainer.args.pad_to_multiple_of=545), label_pad_token_id=-100, padding_side='right'
mapping: reaches the trainer; objective all pads to 545, assistant_only to 537, asked 545
```
