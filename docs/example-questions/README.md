# Example questions

These files show what a respondent model is asked in the [hypothetical-lab study](../../runbook.md#hypothetical-frontier-lab-study): one scenario's three initial questions on the forecast's median trajectory, and what changes for the lab's not-yet-public models and on the fastest trajectory. They are rendered from the committed study inputs and hold no answers.

| File | What it shows |
| --- | --- |
| [`direct.txt`](direct.txt) | The direct question for the scenario `output_reliance_software` about the lab's public models: the expected annual count, the loss of one incident, the two large-loss counts, the loss components, the lab's own loss and the expected annual societal harm. It quotes the scenario's source excerpts and then its [threat model](../threat-models.md). |
| [`chain.txt`](chain.txt) | The event-chain question for the same scenario, asked in its own call: the same count built from an opportunity rate and stage probabilities. |
| [`parent.txt`](parent.txt) | The question about the scenario's taxonomy category, `n_failure_output_reliance`: the expected annual harm of the whole category and of the part outside its scenarios, asked with the excerpts that its scenarios' questions show and the summary of each scenario's threat model. |
| [`not-yet-public.diff`](not-yet-public.diff) | What changes in `direct.txt` when the same question is about the lab's not-yet-public models. Every scenario is asked about both populations. |
| [`fastest-trajectory.diff`](fastest-trajectory.diff) | What changes in `direct.txt` on the fastest of the study's 15 trajectories: two rows of the trajectory table, the doublings that the frontier's rise would be, and the deployment scales. |

Each `.txt` file is one call: its system prompt, then its user prompt. Paragraphs are wrapped at 100 columns for reading, and a respondent receives each as one line; nothing else differs from the text sent. The answer schema that ends each question is one line of JSON. The two `.diff` files are unified diffs of the wrapped text, without context lines.

A collection asks every scenario's direct and chain question and every category's question of each respondent, at each trajectory node and for both populations ([runbook, section 2](../../runbook.md#2-context-bound-labdeployment-questions)). The follow-up calls are not shown, because each quotes a respondent's earlier answers: the retry after a large-loss count that contradicts the stated loss quantiles, the event-chain guardrail retry and the cross-check of the direct and built-up expected harm ([methodology](../../methodology.md#event-populations-and-frequency)). A dry run of the [study script](../../runbook.md#running-the-hypothetical-lab-study) freezes every initial question of the study with its exact text, in each collection's `calls/*/request.json`.

## Regenerating

[`tests/test_example_questions.py`](../../tests/test_example_questions.py) plans the study from the committed inputs and compares these files with what it renders, so a change to the question text, the scenario, its evidence, the hypothetical lab, the forecast or the Epoch table shows up as a diff of these files. Regenerate them only for such an intended change. From a clean checkout, with a new output directory outside it:

```sh
uv run --frozen python scripts/run_hypothetical_lab_study.py ../hfl-dry-run --dry-run
uv run --frozen python scripts/render_example_questions.py ../hfl-dry-run/public-study
```
