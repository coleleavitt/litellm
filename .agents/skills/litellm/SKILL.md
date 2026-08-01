```markdown
# litellm Development Patterns

> Auto-generated skill from repository analysis

## Overview

This skill teaches you the core development patterns, coding conventions, and collaborative workflows for contributing to the `litellm` codebase. `litellm` is a Python-based project (no framework detected) with a significant dashboard UI component. The repository emphasizes strong typing, disciplined code structure, and robust testing practices. It features regular enhancements to both backend API endpoints and frontend UI tables/lists, as well as disciplined dependency management and conflict-resolving merges.

## Coding Conventions

### File Naming

- Python files use `snake_case`, e.g., `management_endpoints.py`, `check_type_discipline.py`.
- Test files follow the pattern `test_*.py`.
- Type definition files are named descriptively, e.g., `management_v1.py`.

### Import Style

- Imports use aliases for clarity and brevity.

  ```python
  import litellm.proxy.management_endpoints.management_v1 as mgmt_v1
  ```

### Export Style

- Named exports are preferred in Python modules.

  ```python
  def get_management_data():
      ...
  ```

### Commit Patterns

- Commit messages use prefixes: `fix`, `feat`, `refactor`, `chore`, `test`.
- Example: `fix: correct budget calculation in management endpoint`

## Workflows

### Add or Update API Endpoint with Contract and Tests

**Trigger:** When adding or updating a backend API endpoint, especially for management entities.  
**Command:** `/new-api-endpoint`

1. Create or update the endpoint implementation in `litellm/proxy/management_endpoints/` or `litellm/proxy/management_endpoints/management_v1/`.
2. Register the endpoint in `__init__.py` or the relevant index file.
3. Update or add types in `litellm/types/proxy/management_endpoints/`.
4. Add or update tests in `tests/test_litellm/proxy/management_endpoints/` or `tests/e2e/management/`.
5. Update UI types in `ui/litellm-dashboard/src/lib/http/schema.d.ts`.

**Example:**
```python
# litellm/proxy/management_endpoints/management_v1/new_endpoint.py
def new_feature_endpoint(request):
    # implementation
    ...
```

### UI Table or List Feature Enhancement

**Trigger:** When improving a dashboard table/list (e.g., Budgets) with new UX features or aligning with backend contracts.  
**Command:** `/ui-table-enhancement`

1. Update or add React components in `ui/litellm-dashboard/src/app/(dashboard)/*/_components/`.
2. Add or update hooks in `ui/litellm-dashboard/src/app/(dashboard)/hooks/`.
3. Update shared table components in `ui/litellm-dashboard/src/components/shared/DataTable/`.
4. Add or update tests for components and hooks.
5. Update types in `ui/litellm-dashboard/src/lib/http/schema.d.ts`.
6. Coordinate with backend API/list contract changes if needed.

**Example:**
```tsx
// ui/litellm-dashboard/src/app/(dashboard)/budgets/_components/BudgetTable.tsx
export function BudgetTable({ data }) {
  // add sorting, filtering, etc.
}
```

### Fix or Refactor Type Discipline and Linting

**Trigger:** When enforcing stricter typing, updating type/lint budgets, or fixing type-related issues.  
**Command:** `/type-discipline-fix`

1. Update `scripts/check_type_discipline.py` or similar scripts.
2. Update `type-discipline-budget.json`, `ruff-strict-budget.json`, or `basedpyright-code-budget.json`.
3. Update or add corresponding tests in `tests/test_litellm/test_check_type_discipline.py`.
4. Re-ratchet or update budgets after merges.

**Example:**
```python
# scripts/check_type_discipline.py
def check_types():
    # logic for type discipline enforcement
```

### Dependency Update with Regression Tests

**Trigger:** When bumping a dependency version to fix a bug or security issue and ensuring coverage with new/updated tests.  
**Command:** `/dep-update`

1. Update the dependency version in `pyproject.toml` or requirements file.
2. Update the lockfile (`uv.lock`).
3. Add or update tests in `tests/local_testing/` or other relevant test directories.
4. Document regression or fix in the commit message.

**Example:**
```toml
# pyproject.toml
aiohttp = "^3.9.0"
```

### Merge Feature Branch with Conflict Resolution

**Trigger:** When merging a large feature or staging branch into the main development branch, especially after parallel development.  
**Command:** `/merge-feature-branch`

1. Merge the remote-tracking branch into the target branch.
2. Resolve conflicts across backend, types, tests, and UI files.
3. Update tests and UI types as needed.
4. Commit with a merge message listing resolved conflicts.

**Example:**
```bash
git merge feature/awesome-feature
# resolve conflicts in .py, .tsx, .d.ts files
git commit -m "merge: resolve conflicts for awesome-feature"
```

## Testing Patterns

- **Framework:** `vitest` is used for UI/TypeScript tests (`*.test.tsx`).
- **Python tests** are located in `tests/`, following the `test_*.py` pattern.
- Tests are closely tied to features and updated alongside code changes.
- Regression tests are added for dependency updates.

**Example:**
```python
# tests/test_litellm/proxy/management_endpoints/management_v1/test_new_endpoint.py
def test_new_feature_endpoint():
    # test implementation
```

## Commands

| Command                | Purpose                                                            |
|------------------------|--------------------------------------------------------------------|
| /new-api-endpoint      | Add or update a backend API endpoint with contracts and tests       |
| /ui-table-enhancement  | Enhance a UI table/list feature and update related contracts/tests  |
| /type-discipline-fix   | Enforce or improve type discipline and linting                     |
| /dep-update            | Update a dependency and add regression tests                       |
| /merge-feature-branch  | Merge a feature branch with conflict resolution                    |
```