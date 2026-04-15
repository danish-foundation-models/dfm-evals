# GitHub Review Guardrails

This repo now supports adding a new local task without editing
`dfm_evals/_exports.py`.

The checked-in pieces are:

- task auto-discovery for modules under `dfm_evals/tasks/`
- `.github/CODEOWNERS` entries for core code and the current task set

To make GitHub enforce the review boundaries, add a branch protection rule or
ruleset for `main` with these settings:

1. Require a pull request before merging.
2. Require review from code owners.
3. Apply the rule to administrators too if you do not want admin bypass.

If you want all pull requests reviewed, also set a blanket approval count such
as `1`.

If you want brand-new task-only pull requests to merge without general review,
do not add a blanket required approval count. In that mode, the code-owner
requirement is what protects the owned paths.

## Task Layout

New task entry points can use any of these layouts:

- `dfm_evals/tasks/my_task.py`
- `dfm_evals/tasks/my_task/task.py`
- `dfm_evals/tasks/my_task/my_task.py`

That is enough for the task to show up through the registry.

## Important Limitation

GitHub does not automatically turn a newly merged task into an owned path.

After you merge a new task and decide it should now count as an "existing task",
add its path to `.github/CODEOWNERS` in a small follow-up change. Keep
`.github/` owned so that update itself still requires review.
