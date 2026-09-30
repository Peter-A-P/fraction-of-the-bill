# The release gate

Project 03's gate decides whether a fine-tune is non-inferior to a frontier model: one suite
per extracted field, a paired bootstrap test on the same 705 post-cutoff filings, a
three-point margin per field, Holm's adjustment across the fifteen, and a power screen. It
grades nothing: this project's grader writes the outcomes (`smallprint gate export`), the
gate reads them through its adapter, and each decision's record names both side files by
their hash. The files, the decisions and the ledger are in [gate-runs/](../gate-runs).
Run on 2026-09-30 with gate 0.1.0.dev2, at its commit `789358c`, the first with the adapter.

| Candidate | Against | Verdict | Fields that block |
|---|---|---|---|
| 7B, seed 0, bf16 | `gpt-5.6-luna` zero-shot, the cost anchor | **pass**, 15 of 15 | none |
| 7B, seed 0, bf16 | `gpt-5.6-sol` two-shot, the best frontier result | block | `net_income` -2.7 (-4.1 to -1.3), `shares_diluted` -1.8 (-3.0 to -0.7) |
| 2B, seed 0, bf16 | `gpt-5.6-luna` zero-shot | block | `operating_income` -3.0 (-4.7 to -1.6), `net_income` -2.0 (-3.8 to -0.3) |
| 2B, seed 0, bf16 | `gpt-5.6-sol` two-shot | block | `operating_income` -3.5 (-5.0 to -2.1), `net_income` -3.4 (-5.0 to -2.0) |
| 4B, seed 0, bf16 | `gpt-5.6-luna` zero-shot | block | `net_income` -2.4 (-4.1 to -0.7) |
| 4B, seed 0, bf16 | `gpt-5.6-sol` two-shot | block | `net_income` -3.8 (-5.5 to -2.3), `cost_of_revenue` -1.8 (-3.1 to -0.6) |

**What this changes.** Averaged over the fifteen fields, both fine-tunes are ahead of the
cost anchor, and the 7B of everything measured. The gate asks a stricter question, whether
any one field could be three points worse, and it answers it field by field. The 7B clears
it against the model it is priced against, on every field. The 2B does not: it is ahead on
eleven fields and level on two, and on operating income and net income it cannot rule out
a three-point loss. It trades those two for large gains on the fields the frontier gets
wrong, cash and state of incorporation. The 4B is blocked on net income alone against the anchor. No fine-tune clears it
against the two-shot ceiling at 35 times the anchor's price, where net income is the field that holds both back.

So the claim the gate supports is narrower than the average: **the 7B is non-inferior to
the cheapest frontier model on every field, at a quarter of its cost per call**; the 2B is
cheaper still and better on average, and worse on two income-statement lines. The gate
decides on seed 0 of each; the other seeds' test runs are in
[training.md](training.md#seeds-the-second-epoch-and-the-formats-2026-09-26).

## Running it again

```bash
smallprint gate export --candidate data/finetuned/<run>-bf16-test_post_cutoff \
  --baseline data/baseline/<frontier run> --out gate-runs/<name> --build-dir data/build/full
# in a checkout of ai-release-gate, with `uv sync --extra gate`:
uv run gate compare --spec gate-runs/<name>/spec.yaml --baseline gate-runs/<name>/baseline.json \
  --candidate gate-runs/<name>/candidate.json --ledger gate-runs/ledger.jsonl
```

It exits 1 on a block. The spec's source kind is the gate's `outcomes_file`; this project
first named one of its own, before the gate had an adapter to read any.
