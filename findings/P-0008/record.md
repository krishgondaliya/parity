# P-0008: On its vision path, TRL trains an empty text message before Inkling-2-Mini tool calls that the template doesn't render

- Status: candidate
- Severity: medium (effect: three extra tokens (an empty text message) are rendered and trained in each affected assistant turn, so the model learns to open an empty text message before its tool call or after its reasoning, which the model's own render never has; reach: assistant turns with tool calls or reasoning whose `content` is the empty string, a common convention in tool-calling and reasoning data, but not reached when `content` is `null`), by effect x reach (docs/DECISIONS.md)
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
- **Documented as intended behavior:** no. TRL documents `prepare_multimodal_messages` as converting string content into structured blocks; nothing says the rendered text may change.
- **Attribution, open:**
  - **TRL's side:** it wraps the empty string as one empty text part instead of no parts.
  - **The model's side:** its template treats an empty text part differently from an empty string.
  - Either fix removes the difference. Rule 4 is cleared for our side; which side to report to is for the skeptic and the human.
- **Duplicates searched** (read-only, TRL issues and PRs): "prepare_multimodal_messages empty content", "prepare_multimodal_messages tool_calls", "empty text block tool call processor", "inkling SFTTrainer", "content empty string tool_calls vlm". No duplicate. Related:
  - #6928 / #6930 (closed): vLLM-side handling of content-less tool calls;
  - #8497 (open): tracking Inkling 2 support, nothing about this;
  - #8513 (merged 2026-11-24, after v1.18.2): an Inkling v1 training template.

## Skeptic verdict
Not run yet (the human asked for it on P-0007).

## Repro
- Not written yet. It needs the release's tokenizer, template and processor (Inkling2Processor, transformers 5.24 with torchvision), and a model class that accepts images.

## Upstream
- Issue URL (posted by human): none
- Maintainer response: none yet
- Fixed in: not yet
