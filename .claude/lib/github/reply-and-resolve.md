# Reply and Resolve

How to answer each of the three feedback surfaces from
[fetch-comments](./fetch-comments.md). Every piece of feedback gets a reply;
only surface A can additionally be resolved.

## Surface A — inline review threads

Both steps are required.

### Step 1: Reply to the comment

```bash
gh api "repos/${PR_REPO_OWNER}/${PR_REPO_NAME}/pulls/${PR_NUMBER}/comments/${COMMENT_DATABASE_ID}/replies" \
  -f body='Fixed — description'
```

**Important:** Always use single quotes for `-f body='...'` to avoid bash history expansion issues with `!` and other special characters.

**Response templates:**

- Fixed: `'Fixed — description'`
- Skip: `'Current code follows .claude/rules/<file>'`
- Acknowledged: `'Acknowledged, thank you!'`

### Step 2: Resolve the thread

Use the thread `id` (the GraphQL node ID from [fetch-comments](./fetch-comments.md), NOT the `databaseId`):

```bash
gh api graphql -f query='
mutation ResolveThread($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) {
    thread { isResolved }
  }
}' \
-f threadId="$THREAD_ID"
```

`$THREAD_ID` is the `id` field from the `reviewThreads.nodes[]` returned by [fetch-comments](./fetch-comments.md).

**Both steps are mandatory.** Do not skip the resolve step.

## Surfaces B and C — review bodies and conversation comments

These have **no resolve mutation** — `resolveReviewThread` accepts only a
review-thread node ID and errors on a review or issue-comment ID. Answer them
with a PR conversation comment and record the ID in the ledger.

```bash
gh pr comment "$PR_NUMBER" --body "@${AUTHOR} Fixed — description"
echo "$NODE_ID" >> "/tmp/pr-${PR_NUMBER}-handled.txt"
```

- **Address the author by `@login`** — a conversation comment is not threaded, so without the mention it is unclear which feedback it answers. Quote the point being answered when several are outstanding.
- **Batch one reply per iteration.** Answering five conversation comments with five separate posts spams the PR; one comment with a short bullet per point is the norm.
- **Recording the ID is what makes the reply stick.** Skip it and the next iteration re-presents the same feedback as unaddressed.
- Feedback that is purely informational (bot walkthrough, "nice work") needs no reply — just record its ID.

Template for a batched reply:

```text
@reviewer Addressed:
- Rebased onto main.
- Null check added on the predicate path.
- Kept the existing enum style — follows .claude/rules/codestyle.md §2.
```
