"""What Unsloth documents about serving modes, importable without Unsloth.

The Unsloth adapter raises ModeNotConfigurable with this text, and the evidence scripts (which render in the core
environment) use it to re-derive saved verdicts. The report card shows it on every such cell.

Decided by the human, 2026-11-25 (P-0006), under the existing rule (2026-11-24: a framework that documents no way to
set a serving mode gets not_configurable cells):
- Unsloth documents non-thinking training only as the soft switch: the recommended Qwen3 notebook appends "/no_think"
  to every non-reasoning user turn, so the template's empty think block is model output the model learns to emit
  (NOTEBOOK, cells 19-20).
- Our non-thinking reference is the hard switch (`enable_thinking=False`): the generation prompt pre-fills the block,
  and the model card says the model then emits none.
- Unsloth documents no way to prepare training data for the hard switch. `enable_thinking=False` in the formatting
  call changes nothing on training text, and its only citation is an inference example.
"""

NOT_CONFIGURABLE = (
    "Unsloth documents non-thinking training via the /no_think soft switch; hard-switch training is not configurable."
)
# A release Unsloth documents no training recipe for (DeepSeek V4.1, the human's decision, 2026-11-26): no documented
# way to prepare training data in another serving mode either.
NOT_CONFIGURABLE_UNDOCUMENTED = (
    "Unsloth documents no training recipe for this model, so no way to set its serving mode in training data."
)
NOTEBOOK = (
    "https://github.com/unslothai/notebooks/blob/ebe88117bb560d296910f6e733693294d630f024/"
    "nb/Qwen3_(14B)-Reasoning-Conversational.ipynb"
)
DECISION = "docs/DECISIONS.md, 2026-11-25 (P-0006): Unsloth's non-default serving modes are not_configurable"
