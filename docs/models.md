# Base models, their licences and their cutoffs

**Not yet chosen. Nothing may be fine-tuned until this is filled in.**

A gate, not a placeholder. Three things have to be recorded here before a training run
starts, and two of them decide whether the project's headline number means anything.

## What goes here

| Column | Why it is not optional |
|---|---|
| Family, exact revision and parameter count | Three sizes, roughly 1.5B, 4B and 8B |
| **Licence**, quoted and linked | Only families whose terms permit publishing adapters *and* merged weights. A base whose licence forbids redistribution cannot be one of the three, however good it is |
| **Documented training cutoff**, with the source | The split depends on it. `assign_splits` takes it as `cutoff`, and the post-cutoff test set is the headline |
| Context length, tokeniser, chat template | The prompt is built against these and the load test runs at the context length recorded here |
| Where the weights were obtained and their digest | So the run is reproducible from the recipe alone |

## The cutoff is the load-bearing one

The headline number is accuracy on filings published after the base model's training
cutoff, and the pre-cutoff set is reported beside it as the contamination gap. If a
model's cutoff is undocumented, or documented only as a month with no day, that has to be
recorded here as the vagueness it is, and the split takes the *later* end of the stated
range so the post-cutoff set stays honest. A base whose cutoff cannot be established at all
is not usable for this project's central claim, and using it anyway while reporting a
post-cutoff number would be the kind of quiet dishonesty the whole repository is arranged
against.

Where the three cutoffs differ, the split uses the latest of them, so that one held-out set
is after every base model's cutoff and all three sizes are measured on identical items.

## Model cards

Every published adapter and every merged or quantised weight file gets a card under
[`cards/`](cards/), in the same commit as the upload. Each card carries the per-field
accuracy table and the named failure modes from `field_report`: how often the model reads
the prior-year column, drops the reporting scale, inverts a loss in parentheses, invents a
value the filing does not report, or fails to emit parseable JSON at all. A card with only
a headline number is not a card.
