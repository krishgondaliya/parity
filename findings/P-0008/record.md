# P-0008: On TRL's vision path, Inkling-2-Mini tool-call turns with empty content gain a trained empty text message

- Status: candidate
- Severity: low, provisionally (effect: three extra tokens (an empty text message) are rendered and trained in each affected assistant turn, which the template's string-form render doesn't have; reach unmeasured: TRL's documented tool-call format leaves `content` out, which becomes null and is unaffected, so only data that writes `content: ""` on tool-call or reasoning turns is reached. It becomes medium if ordinary datasets turn out to do that), by effect x reach (docs/DECISIONS.md)
- Date found: 2026-11-26
- Model: thinkingmachines/Inkling-2-Mini@7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6 (the release's own architecture and processor)
- Framework: trl@1.18.2, `vlm` extra   (lockfile: envs/trl-1.18.2/vlm/uv.lock)
- Check: t1.template_parity   Fixture: tool_call   Kind: bug
- Verdict file: docs/evidence/2026-11-26-m5-trl-inkling/verdicts/thinking/t1.template_parity__tool_call.json
- Predicted before the run: yes (`docs/evidence/2026-11-26-m5-trl-inkling/predictions.md`, from reading TRL's source and Inkling's template).

## Evidence
- **The M5 verdict:** first divergence at token 158. After the tool-call turn's `<|message_model|>`, Inkling's own render continues `get_weather<|content_invoke_tool_json|>{…`; TRL's row continues `<|content_text|><|end_message|><|message_model|>get_weather…` (251 tokens, against 248). The other five checks on that fixture are inconclusive: the rows don't line up with the template's render.
- **Where it appears** (`docs/evidence/2026-11-26-m5-trl-inkling/triage_tool_call.py` and `.json`). TRL's own `SFTTrainer`, `assistant_only_loss=True`; each row compared with Inkling's own render of the same messages:

  | Assistant turn | TRL's vision path (`Inkling2Processor`) | TRL's text path (tokenizer) |
  | --- | --- | --- |
  | tool call, `content: ""` (the fixture) | +3 tokens | same as the template |
  | tool call, `content: null` | same | same |
  | tool call, with text content | same | same |
  | tool call, `content: ""`, with reasoning | +3 | same |
  | two tool-call turns, `content: ""` | +6 (3 per turn) | same |
  | reasoning only, `content: ""` | +3 | same |
  | empty reply, `content: ""`, no tool call or reasoning | same | same |

## Mechanism (TRL 1.18.2; installed files byte-identical to the tag)
1. A model id builds `config.architectures[0]`, which for Inkling-2-Mini is `Inkling2LoopForConditionalGeneration` (`trl/trainer/utils.py` L1362-L1375). That model accepts images, so `SFTTrainer` loads `AutoProcessor`, here `Inkling2Processor` (`sft_trainer.py` L714-L718). This is the path a text-only fine-tune of this release takes in TRL.
2. With a processor, `_tokenize` first calls `prepare_multimodal_messages` (`trl/data_utils.py` L367 and below). That wraps every string `content`, the empty string included, as `[{"type": "text", "text": …}]`.
3. Inkling's template renders a string `content` as a text message unless it is empty on an assistant turn with tool calls or reasoning (`chat_template.jinja` L75-L76). A list `content` renders one text message per part, even an empty one (L77-L83).
4. So an assistant turn with `content: ""` and tool calls or reasoning gains an empty text message, inside its `{% generation %}` block, so it is trained. `content: null` isn't wrapped, so it renders as the template does.

## Triage (from /triage)
- **Calibration green on the known-good model:** yes (M4, and in the `vlm` variant 96/96, `docs/evidence/2026-11-26-m4-recalibration/`).
- **Adapter uses the documented public path and option:** yes. A tiny release-architecture model and the processing class TRL's own `model_accepts_images` picks, `SFTConfig(assistant_only_loss=True)`. A user passing the model id gets the same model class and processor.
- **Fixture valid for this model:** yes. An assistant turn with `content: ""` and `tool_calls` is legal OpenAI-format data.
- **Reference consistent:** yes. Our reference is Inkling's own template, with the messages as given (string content), which is also what the tokenizer and the template's own markers give; our preview found the markers agree with our reference on all 8 fixtures.
- **Thinking Machines' official renderer agrees with it** (2026-11-26; tml-renderers 0.5.0, the reference for P-0008 by the human's rule, `docs/DECISIONS.md`; run sandboxed, `docs/evidence/2026-11-26-p0008-renderer/`):
  - with the template's default thinking effort (0.9), `renderers.Inkling2().render_for_sft` gives exactly the Hugging Face template's text and token ids for the `tool_call` fixture with the tool-call turn's content `""`, null or absent;
  - it renders no empty text message for any of the four forms, including one empty text part (`[{"type": "text", "text": ""}]`). The Hugging Face template renders one for that form, and TRL's vision-path row has one for `""`;
  - so TRL's row for `""` differs from the model's official format by those three trained tokens, and the template's list branch differs from the model's own renderer.
- **Documented as intended behavior:** partly. TRL's wrapping of `""` as `[{"type": "text", "text": ""}]` is deliberate (PR #5496, merged 2026-04-10: several VLM processors crash on `content: ""` but accept the empty text part), and TRL's own internals pass `content: ""` with tool calls. Its effect on Inkling 2's render is not intended or documented.
- **Attribution (after the skeptic):** mainly the model's template.
  - Inkling 2's template contradicts itself: its string branch skips an empty text message on assistant turns with tool calls or reasoning (L75-L76), and its list branch doesn't (L77-L83). The tokenizer alone gives the same 162-token row from the wrapped messages, so Inkling2Processor plays no part.
  - Inkling v1's template renders the empty text message in both forms. Inkling 2 added the rule for the string form only, which suggests its intended format has none, so the string-form reference is probably right.
  - TRL's side is constrained: not wrapping `""` would bring back #5496's crashes. A TRL training template for Inkling 2 is the realistic TRL-side option.
  - **Decided by the official renderer (2026-11-26):** it never renders an empty text message here, so the template's list branch is what departs from the model's own format. TRL's wrapping of `""` is the trigger.
- **Duplicates searched** (read-only, TRL issues and PRs): "prepare_multimodal_messages empty content", "prepare_multimodal_messages tool_calls", "empty text block tool call processor", "inkling SFTTrainer", "content empty string tool_calls vlm". No duplicate. Related:
  - #6928 / #6930 (closed): vLLM-side handling of content-less tool calls;
  - #8497 (open): tracking Inkling 2 support, nothing about this;
  - #8513 (merged 2026-11-24, after v1.18.2): an Inkling v1 training template.

## Skeptic verdict (2026-11-26)
- **Real candidate, partly:** the behavior is real and not our harness, but it isn't a TRL bug on its own. The template contradicts itself, and TRL's wrapping that triggers it is deliberate. Confidence high on the reproduction, the mechanism and the labels; medium on which side owns it.
- **Checked by the skeptic:**
  - the tokenizer and processor renders of unwrapped string content are equal;
  - the wrapped messages give 162 tokens against 159, with the tokenizer alone too;
  - `content: []` renders like `""`;
  - the inserted tokens have `assistant_masks` 1, and their labels equal their ids, in the dataset and after collation;
  - TRL loading the processor itself (`processing_class=None`) gives the adapter's row;
  - `prepare_multimodal_messages` and `_tokenize` are unchanged on TRL main;
  - no duplicate in TRL or in the model repo's discussions.
- **Corrections, applied:** severity provisionally low (reach unmeasured; the "common convention" wording dropped); "the template's string-form render" instead of "the model's own render"; the wrapping is deliberate (#5496); TRL's default path for a model id, while transformers documents `Inkling2LoopForCausalLM` for text-only use, which takes the unaffected tokenizer path.
- **To check before any upstream draft:**
  1. **Done 2026-11-26:** the official renderer, run under the human's conditions after the human's OK (the skeptic's earlier attempt was blocked and not re-run on its behalf). It renders no empty text message in any form and equals the Hugging Face template for `""`, null and absent (above).
  2. What vLLM and SGLang render for a history turn with `""` plus tool calls.
  3. Whether `[]`, or leaving `""` alone on tool-call turns, avoids #5496's crashes (whether any TRL-side fix is possible).
  4. How often popular tool-calling datasets write `""` rather than null on tool-call turns.
- **Report to:** the model repo first (its list branch should skip an empty text part the way its string branch does), and TRL only as a note on #8497 or a request for an Inkling 2 training template.

## Repro
- **`findings/P-0008/repro.py`** (2026-11-26), 30 lines, no Parity code, tokenizer and template files only, aimed at the model repo. Inkling-2-Mini's template on one tool-call conversation, the tool-call turn's content as `""`, null and one empty text part. It prints the official renderer's expectation (no empty text message for any form) and the counts (0, 0 and 1). An optional last line, when TRL is installed, shows `prepare_multimodal_messages` turning `""` into the list form. Exit 1.
- **Runs:** 3 of 3 exit 1 in a fresh venv (transformers 5.24.0, trl 1.18.2, torch 2.15.0, resolved against Parity's TRL `vlm` lock), and exit 1 with TRL made unimportable. There's no fixed version to test: the template and tokenizer files are byte-identical at the model's main (`c9d0283`), and `prepare_multimodal_messages` is identical on TRL's main (`c2d5ff4`). Commands and expected output: `findings/P-0008/HOW-TO-RUN.md`; lockfile `findings/P-0008/uv.lock`.

- **Repro, after the issue-reviewer (2026-11-27):** jinja2 pinned in its comment (transformers alone raises `ImportError` in `apply_chat_template`, which also exits 1), and the unneeded `copy` removed. 3 of 3 exit 1 in the pinned venv with identical output; with only transformers and jinja2, the four template lines and exit 1.
- **Renderer on the repro's own conversation** (`docs/evidence/2026-11-26-p0008-renderer/repro_cases.py`, `repro_results.json`, the same sandbox, 2026-11-27): the four forms accepted; the same output for each; equal to the template's text and token ids for `""`, null and missing; no empty text message.
- **The suggested template change** (`findings/P-0008/fix_check.py`, `fix_check.json`): with it, the template equals the renderer on all four forms, on the fixture and on the repro's conversation; 18 of 19 other renders unchanged; the one that changes is TRL's row for `tool_call` (every string content as one text part), which then equals the template's render of the fixture as given.
- **Same template elsewhere** (Hub API, read-only, 2026-11-27): `chat_template.jinja` has blob `cae99e08917f` at the current main of Inkling-2-Mini (`c9d0283`), Inkling-2-Small (`885d78b`) and Inkling-2-Small-MXFP8 (`14f2cb2`), the same as at the pinned revision.
- **Duplicates on the Hub** (read-only, 2026-11-27): the Community discussions of Inkling-2-Mini (61), Inkling-2-Small (16) and Inkling-2-Small-MXFP8 (9). No duplicate: the closest are Mini #21 (the model's own tool-call behavior) and the merged PRs that added generation markers to the Small repos' templates.

## Upstream
- Issue URL (posted by human): not posted; draft in findings/P-0008/issue.md (a Community discussion for the model repo; reviewed by the issue-reviewer, 2026-11-27)
- Maintainer response: none yet
- Fixed in: not yet
