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

## 4. Two axes, and why the arch axis already implies the other

| Axis | Output | Used by |
| ---- | ------ | ------- |
| non-code | `non_code_only` | arch-agnostic jobs: `ut`, `packaging-matrix`, `ut-a2a3`, `ut-a5` |
| architecture | `a2a3_changed` / `a5_changed` | arch-specific jobs: `st-sim-*`, `st-onboard-*`, `profiling-flags-smoke` |

These are layered, not redundant. `a2a3_changed` and `a5_changed` are computed
by subtracting `NON_CODE` *first*, so a non-code-only change already makes both
false. An arch-gated job therefore needs no separate non-code check, and adding
one would be noise. An arch-agnostic job cannot use the arch axis at all.

Keep that asymmetry deliberate. If a new job is arch-specific, gate it on the
arch axis; if not, on `non_code_only`. Mixing them per-job is how §1 gets
violated again.

## 5. Fail open once, at the top

An empty file list means attribution is impossible — a PR with no files, or
base/head SHAs that did not resolve. Every flag must then report "affected",
and **that decision belongs in one guard before any pattern runs**, not inside
each flag's branch.

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
