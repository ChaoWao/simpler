# CI Change Detection and Job Gating

Every *downstream* PR job in `.github/workflows/ci.yml` is gated by an output
of the `detect-changes` job. Two are deliberately ungated: `detect-changes`
itself, which produces the outputs, and `pre-commit`, which every gated job
declares in `needs:` and which must therefore always run. Those gates decide
whether a change occupies four self-hosted NPU machines for ten minutes or
finishes in ninety seconds on a GitHub runner. They are cheap to get subtly wrong and the symptom is silent —
a job that should have skipped merely *passes*, so nobody notices until a
flake in it reddens an unrelated PR.

This rule is about keeping the gates honest. It is not a description of the
current filters; those change. It is the set of invariants they must satisfy.

## 1. One vocabulary for "cannot change what the code does"

`detect-changes` defines exactly one such set, `NON_CODE`. **Every gate derives
from it.** Do not introduce a second, narrower notion for some jobs and a wider
one for others.

The concrete failure this prevents: the file set once had two definitions —
`NON_CODE` (which covered `docs/`, `.claude/`, `.gitignore`, `*.md`) and a
separate `docs_only` meaning *every file ends in `.md`*. Arch-specific jobs
gated on the wide one, and the **self-hosted hardware unit tests gated on the
narrow one**. A PR touching only `.gitignore` therefore skipped every scene
test and still booked `ut-a2a3` and `ut-a5` on real silicon.

If you need a second signal, derive it from `NON_CODE` rather than writing a
fresh pattern next to it.

## 2. Membership follows a file's effect, not its path

A file belongs in `NON_CODE` when changing it cannot change what the code does
— not when it happens to live under `docs/`.

- `mkdocs.yml` sits at the repo root and is pure docs tooling. **In.**
- `.github/workflows/docs.yml` is a workflow, and is also pure docs tooling —
  and it runs unconditionally on every PR, so it is its own gate. **In.**
- `.github/workflows/ci.yml` defines the gates themselves. **Out, always.** A
  change to the gating must run everything, including the jobs it might have
  just switched off.

The corollary is a habit, not a pattern: **adding a top-level config file or a
tooling workflow is a change-detection event.** Ask whether `NON_CODE` needs to
learn about it in the same commit. `mkdocs.yml` was added without that check
and for weeks every docs PR that touched it ran the full a2a3 + a5 hardware
matrix.

## 3. Never gate an expensive job on a weaker signal than a cheap one

Rank the gates by how much they let through. A self-hosted hardware job must
never sit behind a *more permissive* condition than a GitHub-hosted one. If you
find yourself writing a looser `if:` for the more expensive job, the vocabulary
is wrong (see §1) — fix that instead of widening the gate.

## 4. Three axes, layered — and the non-code one is subtracted first

| Axis | Output | Answers |
| ---- | ------ | ------- |
| non-code | `non_code_only` | can this diff change what the code does at all? |
| architecture | `a2a3_changed` / `a5_changed` | which silicon can it reach? |
| test category | `st_affected` / `ut_affected` | which suite can it break? |
| example corpus | `examples_only` | can it reach a job that builds the product and runs a fixed payload? |

They are layered, not redundant. **Every arch and category flag subtracts
`NON_CODE` before deciding**, so a non-code-only change already makes all four
false. An arch- or category-gated job therefore needs no separate non-code
check, and adding one would be noise.

A job composes the axes it is actually subject to:

| Job family | Gate |
| ---------- | ---- |
| `st-sim-*`, `st-onboard-*` | `<arch>_changed && st_affected` |
| `profiling-flags-smoke` | `(a2a3_changed \|\| a5_changed) && !examples_only` |
| `ut`, `ut-a2a3`, `ut-a5` | `non_code_only != true && ut_affected` |
| `packaging-matrix` | `non_code_only != true && !examples_only` |

`packaging-matrix` and `profiling-flags-smoke` build and install the product and
then exercise it with a **fixed, tiny payload** — one entry-point script and one
`vector_example` respectively. They are the only jobs that consume neither test
suite as a corpus, which is why they take the example axis rather than the
category one. Note what stays in: `tests/` still triggers packaging, because
`tools/verify_packaging.sh` runs a `tests/st/` file as its entry-point smoke and
`tests/` is in the sdist `include`. Only `examples/` is provably absent from
both jobs.

The UT jobs stay off the arch axis on purpose — unit tests cover shared
contracts, so the cost of a falsely-skipped regression outweighs the minutes.
That is a decision about the *arch* axis only; the category axis is a different
question, because a scene-test-only change genuinely cannot break a unit test.

### Write every axis in the same shape

The arch and category axes use one pattern: **a partition is unaffected only
when every changed file belongs exclusively to a sibling partition.**

```bash
SIBLING='^(...)'                     # what this partition is NOT
REMAINING=$(echo "$FILES" | grep -vE "$SIBLING" | grep -vE "$NON_CODE" || true)
[ -n "$REMAINING" ] && flag=true || flag=false
```

Two properties come for free, and both are why a new axis must copy the shape
rather than invent one:

- **Fail-safe direction.** Any path the patterns do not recognise survives the
  filters, so it lands in `REMAINING` and turns the flag *on*. New directories
  over-run CI rather than silently skipping it.
- **Shared infrastructure resolves correctly without being enumerated.** The
  root `conftest.py`, `pyproject.toml`, `simpler_setup/` and `tests/lint/`
  belong to no single partition, so they match no `SIBLING` and flip every flag
  true — which is what they should do, since both suites load them.

## 5. Fail open once, at the top

No usable file list means attribution is impossible — a PR with no files, or a
`git diff` that failed on unresolved base/head SHAs. Every flag must then report
"affected", and **that decision belongs in one guard before any pattern runs**,
not inside each flag's branch.

"No usable list" covers both cases, and the failure one is the easier to miss.
`run:` is `bash -e`, so a non-zero `git diff` aborts the step before any guard —
and a failed `detect-changes` leaves every downstream `needs:` unsatisfied, which
**skips** the matrix instead of running it. That is fail-closed, the opposite of
the intent. Let only the exit status decide and fold a failure into the empty
case:

```bash
if ! FILES=$(git diff --name-only "$BASE"..."$HEAD"); then
  FILES=""
fi
```

Deciding it per-flag is how the axes drift apart. The emptiness test once lived
only inside the `non_code_only` branch, so an empty diff left that flag `false`
— UT and packaging ran — while every arch flag independently came out `false`
too, skipping the scene tests. Two axes failing in opposite directions on the
same input, with a doc claiming it "runs the full matrix".

The same applies to any new axis: add its outputs to the existing guard rather
than re-testing emptiness alongside its own pattern.

## 6. A schedule trigger belongs in its own workflow

`sanitizers.yml` is separate from `ci.yml` specifically so its `cron` fires
only its own jobs. Adding `schedule:` to `ci.yml` would run every ungated job
— `pre-commit`, `ut`, `packaging-matrix`, and both self-hosted hardware pools —
nightly. Same for any future trigger that is not `pull_request` or a `push` to
the default branch.

## 7. Verify by observation, never by reading the regex

A shell `grep -vE` chain over a file list is not reviewable by inspection; the
`.md$` alternation binding to the whole pattern rather than one branch is the
kind of detail that reads fine and behaves otherwise. After changing any gate:

```bash
# What actually ran, on a PR whose diff you know:
gh pr checks <NUMBER> --repo hw-native-sys/simpler
```

Confirm the jobs you expected to skip report `skipping`, not `pass`. **A gating
bug shows up as a green check, so "CI passed" is not evidence.** Where the
change is to `NON_CODE` itself, replay the pattern locally against
representative file lists — a docs-only set, a `.gitignore`-only set, a
single-arch set, a `ci.yml` set — and check all four land where you intended
before pushing.

## Relation to the other rules

- [`discipline.md`](discipline.md) §5 makes every red check on your PR yours to
  triage. This rule covers the inverse: a check that ran when it should not
  have is also a defect, and it is invisible precisely because it is green.
- [`doc-consistency.md`](doc-consistency.md) §1 asks you to grep for references
  after a rename. Renaming a `detect-changes` output is exactly that case —
  every `needs.detect-changes.outputs.<name>` consumer moves in the same
  commit.
