# P-0002: Axolotl never trains the closing token of an empty assistant turn, despite `train_on_eos: turn`

- Status: candidate
- Date found: 2026-11-24
- Model: Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca
- Framework: axolotl@0.21.1   (lockfile: findings/P-0002/uv.lock, a copy of envs/axolotl-0.21.1/uv.lock; findings/P-0002/requirements.txt is the same resolution without Parity)
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
- **Shrunk** (`/shrink-repro`, 2026-11-24): `findings/P-0002/repro.py`, 26 lines, no Parity code. It builds the strategy with `axolotl.prompt_strategies.chat_template.load` and calls `tokenize_prompt`, which is the code `type: chat_template` runs and the entry point Axolotl's own tests use. It downloads Qwen3-0.6B tokenizer and config files only (no weights), runs in a few seconds on CPU, and exits 1 when the empty turn's `<|im_end|>` is unlabeled.
  - Conversation: `[user, assistant "Hi.", user, assistant ""]`, an empty *last* turn. Expected trained `<|im_end|>`s `[False, True, False, True]`; actual `[False, True, False, False]`. No token of the empty turn has a label.
  - Strategy-level labels are identical, input ids and labels, to the full `load_datasets` path that `axolotl preprocess` runs, on every variant below (checked).
- **Fresh environment, pinned versions** (`uv venv` + `uv pip install -r requirements.txt`, Parity not importable): fails 3/3 (exit 1), both before and after the issue reviewer's shortening. axolotl 0.21.1, transformers 5.20.1, tokenizers 0.23.3, torch 2.14.1, Python 3.12.13.
- **Main** (`axolotl @ git+…@de8549a4406327ab5311f7187a8afaa232645487`, 2026-11-20, resolves to 0.21.2.dev0 with transformers 5.22.0 and torch 2.15.0, fresh env): fails 3/3, same output. Not fixed.
  - No release after v0.21.1. Main is 69 commits ahead; three touch `chat_template.py` (a998b3d34c, 8b51899b0d, 5fe10dc541): an import, type hints and docstrings, and the `chat_template_kwargs` conversion. The labeling loop (`_tokenize_single_prompt` through `find_first_eos_token`) is byte-identical on main.
  - No open PR or issue covers it (searched PRs and issues: "empty turn", "empty assistant", "train_on_eos", "find_turn", "empty content eos"; the issue reviewer ran 10 more queries). Open PR #4132 changes the turn locator, but only after a span is found; the `char_end <= char_start` return that empty turns take is untouched (checked).
- **Only log line:** an unrelated torchao `SyntaxWarning`. Nothing mentions the empty turn.

## Variants tested (2026-11-24, strategy level and full preprocess, identical)
| Case | Empty turn's `<|im_end|>` trained? |
| --- | --- |
| Empty content, history turn | no (P-0002) |
| Empty content, last turn | no, and no token of the turn is trained |
| Tool call with `content: ""`, history turn | yes (tool call and `<|im_end|>` trained) |
| Tool call with `content: None`, history turn | yes |
| Tool call with `content: ""`, last turn | yes |
| Tool call with `content: None`, last turn | yes |
| `reasoning_content` with `content: ""`, history turn | no (Qwen3 drops history reasoning, so the turn renders empty) |
| `reasoning_content` with `content: ""`, last turn | yes (the reasoning field anchors the span) |

- **Severity: stays low.** Tool-use data is not affected. Turns whose only content is a tool call keep their span and end token. What is affected: assistant turns that render empty. Those are a deliberately empty reply, or, with Qwen3, a history turn whose only content was reasoning. Each loses one token of loss (its end token), and an empty last turn loses all of its loss, silently.
- A one-turn conversation with an empty reply has no trainable tokens at all, so Axolotl drops the row with a warning (`utils/trainer.py` L263-313). That case is loud, not silent.
- Mechanism, re-checked at v0.21.1 (main is the same, lines +4): `_find_turn_from_text` returns `(-1, -1)` for an empty turn (`chat_template.py` L966-967), so `find_turn` returns `(-1, -1)` (L1146-1147). Content labeling is skipped (L738), and the EOS search runs from `start_idx=-1` (L800). It finds an earlier `<|im_end|>` that fails `abs(token_idx - turn_end_idx) <= 3` (L803), so the turn's own end token is never labeled.

## Upstream
- Draft: `findings/P-0002/issue.md`, laid out for Axolotl's issue form (blank issues are disabled there). Not posted; the human runs the repro, then posts.
- Reviewed by the `issue-reviewer` subagent (2026-11-24): "ready to post after the fixes". Applied:
  - the form layout and the title in title casing;
  - the script's exact stdout as expected vs actual;
  - two accuracy fixes: the empty turn isn't the model's only output (Qwen3 adds an empty think block), and the Axolotl thinking test doesn't cover Qwen3's empty block;
  - "No warning is logged" instead of "Nothing is logged" (an INFO line prints);
  - torch versions;
  - the shorter repro (re-run 3/3 on both fresh environments);
  - the middle-turn output and the concrete EOT/EOS arithmetic (both checked by me).
