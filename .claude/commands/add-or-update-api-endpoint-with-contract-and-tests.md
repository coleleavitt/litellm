---
name: add-or-update-api-endpoint-with-contract-and-tests
description: Workflow command scaffold for add-or-update-api-endpoint-with-contract-and-tests in litellm.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /add-or-update-api-endpoint-with-contract-and-tests

Use this workflow when working on **add-or-update-api-endpoint-with-contract-and-tests** in `litellm`.

## Goal

Implements or updates a backend API endpoint, registers it in the proxy, updates types/contracts, and adds/updates corresponding tests and UI schema types.

## Common Files

- `litellm/proxy/management_endpoints/management_v1/*.py`
- `litellm/proxy/management_endpoints/management_v1/__init__.py`
- `litellm/types/proxy/management_endpoints/management_v1.py`
- `tests/test_litellm/proxy/management_endpoints/management_v1/*.py`
- `tests/e2e/management/*.py`
- `ui/litellm-dashboard/src/lib/http/schema.d.ts`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Create or update endpoint implementation in litellm/proxy/management_endpoints/ or litellm/proxy/management_endpoints/management_v1/
- Register endpoint in __init__.py or relevant index file
- Update or add types in litellm/types/proxy/management_endpoints/
- Update or add tests in tests/test_litellm/proxy/management_endpoints/ or tests/e2e/management/
- Update UI types in ui/litellm-dashboard/src/lib/http/schema.d.ts

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.