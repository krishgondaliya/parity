# P-0008's reference: tml-renderers on empty tool-call content

tml-renderers 0.5.0 (module sha256 86e0c106580a…), effort 0.9, sandbox return code 0, temp dir left [], write probe outside the temp dir created: False.

| Tool-call turn's content | Renderer | Empty text messages: renderer / HF template / TRL vision row | Renderer = HF template (text, ids) | Renderer = TRL vision row |
| --- | --- | --- | --- | --- |
| empty string | renders | 0 / 0 / 1 | True, True | False |
| null | renders | 0 / 0 / 0 | True, True | True |
| absent | renders | 0 / 0 / 0 | True, True | True |
| one empty text part | renders | 0 / 1 / 1 | False, False | False |
