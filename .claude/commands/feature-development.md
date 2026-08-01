---
name: feature-development
description: Workflow command scaffold for feature-development in litellm.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /feature-development

Use this workflow when working on **feature-development** in `litellm`.

## Goal

Standard feature implementation workflow

## Common Files

- `ui/litellm-dashboard/src/app/(dashboard)/budgets/_components/*`
- `ui/litellm-dashboard/src/app/(dashboard)/hooks/budgets/*`
- `ui/litellm-dashboard/src/app/(dashboard)/hooks/common/*`
- `**/*.test.*`
- `**/api/**`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Add feature implementation
- Add tests for feature
- Update documentation

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.