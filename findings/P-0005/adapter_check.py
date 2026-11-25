"""P-0005, rule 4: does our adapter's pad_to mapping reach Unsloth's trainer? Run it on the GPU route (the Colab kit).

Runs Parity's Unsloth adapter itself, not the repro, on the batch t1.special_tokens builds for single_turn: single_turn
with long_answer, padded to pad_to = the longer row + 8.
- objective "all": the recommended notebook without train_on_responses_only, so the trainer keeps its own collator;
- objective "assistant_only": the recipe, with train_on_responses_only.
Each prints the batch width and the adapter's live declaration of the collator: its pad_to_multiple_of,
trainer.args.pad_to_multiple_of and the padding side.
If "all" pads to pad_to and "assistant_only" doesn't, the adapter's mapping reaches the trainer and the setting is
lost in train_on_responses_only (P-0005). Exit 0 if "all" pads to a multiple of pad_to; exit 1 if it doesn't, which
makes the adapter's mapping the suspect (rule 4).
"""
import sys

from parity.adapters.unsloth import BACKEND, UnslothAdapter  # Unsloth first, as its notebooks do

from parity.adapters.base import TrainCfg  # noqa: E402
from parity.fixtures import load_fixture  # noqa: E402
from parity.release import load_release  # noqa: E402
from parity.targets import CALIBRATION  # noqa: E402

adapter = UnslothAdapter(load_release(CALIBRATION))
convs = [load_fixture("single_turn"), load_fixture("long_answer")]
lengths = [len(adapter.encode(c, TrainCfg(objective="all")).input_ids) for c in convs]
pad_to = max(lengths) + 8
print(f"backend {BACKEND}; unpadded rows {lengths}; pad_to={pad_to} (the longer row + 8, as t1.special_tokens asks)")
widths = {}
for objective in ("all", "assistant_only"):
    cfg = TrainCfg(objective=objective, pad_to=pad_to)
    widths[objective] = len(adapter.encode_batch(convs, cfg).rows[0].input_ids)
    collator = next(s for s in adapter.declare(cfg).sources if s.startswith("pad_token_id:"))
    print(f"objective {objective}: batch width {widths[objective]}; {collator}")
reaches = widths["all"] % pad_to == 0
print(f"mapping: {'reaches the trainer' if reaches else 'does NOT reach the trainer (adapter suspect)'}; "
      f"objective all pads to {widths['all']}, assistant_only to {widths['assistant_only']}, asked {pad_to}")
sys.exit(0 if reaches else 1)
