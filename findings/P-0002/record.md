# P-0002: Axolotl never trains the closing token of an empty assistant turn, despite `train_on_eos: turn`

- Status: candidate
- Date found: 2026-11-24
- Model: Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca
- Framework: axolotl@0.21.1   (lockfile: envs/axolotl-0.21.1/uv.lock)
- Check: t1.end_of_turn, t1.special_tokens   Fixture: empty_assistant   Kind: bug
- Verdict files: docs/evidence/2026-11-24-m3-axolotl/verdicts/t1.end_of_turn__empty_assistant.json, t1.special_tokens__empty_assistant.json
- Predicted from Axolotl's docs before the run: no (predicted pass)

## Evidence
- Conversation: `[user, assistant(content ""), user, assistant("OK.")]`. Config: `type: chat_template`, `roles_to_train: ["assistant"]`, `train_on_eos: turn`, Qwen3's own template.
- Expected: the empty turn renders `<|im_start|>assistant\n<|im_end|>\n`. Its `<|im_end|>` (position 14) is the only token the model writes for that turn, so it must be trained.
  - The docs promise this: `train_on_eos: turn` trains "the EOS token at the end of each trainable turn" (axolotl/utils/schemas/datasets.py @ v0.21.1), and `roles_to_train: ["assistant"]` makes the turn trainable.
- Got: label −100 at position 14. The second assistant turn's `<|im_end|>` (position 37) is trained. Seen in the row and in the padded batch.
- Mechanism (axolotl/prompt_strategies/chat_template.py @ v0.21.1):
  - With a fast tokenizer, `_find_turn_from_text` returns `(-1, -1)` for the empty turn (lines 966-967), and `find_turn` returns at lines 1146-1147.
  - The EOS search then runs `find_first_eos_token(ids, start_idx=-1)`. It scans `ids[-1]` first, then from 0, and finds the user turn's `<|im_end|>`. That fails `abs(token_idx - turn_end_idx) <= 3` (lines 802-803), so the turn's closing token is never labeled.
  - **Completely silent.** The "This is likely an empty turn" warning (lines 1155-1159) is on a fallback path this case never reaches. The repro's full log has no empty-turn line (checked 2026-11-24), and the loss looks normal.

## Triage (from /triage)
- **Calibration green on known-good model:** this run IS the calibration model.
  - `t1.end_of_turn` and `t1.special_tokens` pass on all 8 fixtures through TRL 1.18.2 and on both test stand-ins. Through Axolotl, they pass on the other 7 fixtures.
  - So the check logic is not the suspect.
- **Adapter uses the documented public path and option:** yes. The real `axolotl preprocess` CLI produced the same row, labels included, as the adapter on `empty_assistant` (position 14 untrained, 37 trained).
- **Fixture valid for this model:** yes. An empty assistant message is a legal message; Qwen3's template renders it as `<|im_start|>assistant\n<|im_end|>\n`. The fixture exists for this case ("Its closing token should still be supervised").
- **Reference consistent:** yes. It agrees with the hand-written golden span (`"<|im_end|>"` for message 1) and with TRL's Qwen3 training template, whose `{% generation %}` block covers the empty turn's `<|im_end|>`.
- **Documented as intended behavior:** no.
  - The docs say the EOS of each trainable turn is trained. The empty turn is trainable (`should_train` is True, `chat_template.py:701`).
  - Nothing documents an exception for empty turns, and nothing is logged.
  - Per the M4 note (`docs/STATUS.md`): an undocumented skip of `empty_assistant` is a candidate finding.
- **Duplicates searched:** `gh search issues --repo axolotl-ai-cloud/axolotl` for "empty turn", "empty turn eos train_on_eos", "empty assistant turn label", "Content end boundary is the same as start boundary", "train_on_eos empty content", "train_on_eos". No duplicate.
  - Related: #3754 (closed; Gemma's turn terminator never trained, a different cause) and #4119 (open; same turn locator, different symptom).

## Skeptic verdict
- **Verdict: real candidate, low severity. Confidence: medium-high.**
- **Strongest argument against:** it only happens on degenerate data and costs one token per turn. The "likely an empty turn" warning shows the authors expected empty turns to have no span. But:
  - no doc says their end token is skipped;
  - `should_train` is True for the turn;
  - Axolotl does train the end token of non-empty history turns, so the behavior is inconsistent.
- **Our side checked:** nothing is wrong in the adapter, the fixture or the reference.
- **Correction, applied above:** the default path is silent; no warning is logged.
- **Other inputs that hit it:**
  - An empty *last* turn: no token of the turn is trained, `<|im_end|>` included. This doesn't depend on thinking mode; in non-thinking mode `<|im_end|>` is the only token to train.
  - A history turn whose content is only a think block, or one with `reasoning_content` and `""` content (Qwen3 strips history reasoning, so it renders empty).
  - Not affected: a tool-call turn with `""` content. Its span covers the tool call, and `<|im_end|>` is trained (verified).
- **Axolotl's tests:** the only empty-turn case (`block_count_mismatch` in `test_chat_templates_generation_tags.py`) asserts only that tagged labels equal plain ones.
- **Not fixed on main:** `chat_template.py` on main differs from v0.21.1 only in imports, type hints and the conversion of `chat_template_kwargs`.
- **No duplicate.** Searched "empty assistant", "empty turn", "empty content eos", "find_turn empty", "turn_end_idx -1" and "train_on_eos". Related: #3754, #4119.
- **What would change its mind:** a maintainer saying empty assistant turns are deliberately left without loss.
- **Next:** `/shrink-repro P-0002`, using an empty *last* turn (independent of thinking mode, and the whole turn goes untrained).

## Repro
- findings/P-0002/repro.py (tokenizer and config files only, telemetry off): exits 1 (reproduces) 3/3 on axolotl 0.21.1.
- On main: not tested.

## Upstream
- Not drafted (comes after the skeptic, `/shrink-repro` and `/draft-issue`; the human posts).
