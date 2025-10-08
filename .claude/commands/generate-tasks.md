---
name: Generate Task List
description: Convert a PRD into a detailed, actionable task list
version: 1.0
tags: [planning, tasks, breakdown]
---

# Generate Task List from PRD

## Goal
To create a detailed, step-by-step task list in Markdown format based on an existing Product Requirements Document (PRD). The task list should guide a developer through implementation.

## Development Guidelines

All work must follow the development guidelines specified in CLAUDE.md:
1. **Write minimal code** - Implement only what is needed to solve the immediate problem, avoiding over-engineering
2. **Prioritize conciseness** - Favor clean, compact solutions over backwards-compatibility when they conflict
3. **Remove legacy code** - Delete unused functions, deprecated methods, and dead code paths immediately
4. **Use descriptive naming** - Variables, functions, and classes should clearly indicate their purpose without requiring comments
5. **Follow standard conventions** - Adhere to PEP 8, type hints, and established Python scientific computing patterns
6. **Document scientific rationale** - Include brief comments explaining the scientific reasoning behind parameter choices, thresholds, and methodological decisions
7. **Validate inputs and outputs** - Always include sanity checks for data dimensions, value ranges, and expected data types to catch analysis errors early
8. **Prioritize reproducibility** - Use fixed random seeds, save processing parameters, and ensure identical inputs produce identical outputs across runs

## Process

### Step 1: Analyze PRD
1. Read and analyze the specified PRD file
2. Extract functional requirements, user stories, and other sections
3. Understand the scope and complexity of the feature

### Step 2: Generate Parent Tasks (Phase 1)
1. Create ~5 high-level parent tasks based on PRD analysis
2. Present these tasks to the user in the specified format (without sub-tasks yet)
3. Inform the user: **"I have generated the high-level tasks based on the PRD. Ready to generate the sub-tasks? Respond with 'Go' to proceed."**
4. **WAIT** for user confirmation

### Step 3: Generate Sub-Tasks (Phase 2)
1. Once the user responds with "Go", break down each parent task into smaller, actionable sub-tasks
2. Ensure sub-tasks logically follow from the parent task
3. Cover all implementation details implied by the PRD

### Step 4: Identify Relevant Files
1. Based on the tasks and PRD, identify potential files that need to be created or modified
2. Include corresponding test files if applicable
3. Provide brief descriptions for each file

### Step 5: Save Task List
Save the generated document as `/tasks/tasks-[prd-file-name].md`

## Output Format

The generated task list **must** follow this structure:

```markdown
## Relevant Files

- `path/to/potential/file1.ts` - Brief description of why this file is relevant
- `path/to/file1.test.ts` - Unit tests for `file1.ts`
- `path/to/another/file.tsx` - Brief description (e.g., API route handler)
- `path/to/another/file.test.tsx` - Unit tests for `another/file.tsx`
- `lib/utils/helpers.ts` - Brief description (e.g., Utility functions)
- `lib/utils/helpers.test.ts` - Unit tests for `helpers.ts`

### Notes

- Unit tests should typically be placed alongside the code files they are testing
- Use appropriate test runner for the project (Jest, Vitest, etc.)

## Tasks

- [ ] 1.0 Parent Task Title
  - [ ] 1.1 Sub-task description 1.1
  - [ ] 1.2 Sub-task description 1.2
- [ ] 2.0 Parent Task Title
  - [ ] 2.1 Sub-task description 2.1
- [ ] 3.0 Parent Task Title (may not require sub-tasks if purely structural)
```

## Interaction Model

The process **explicitly requires** a pause after generating parent tasks to get user confirmation ("Go") before proceeding to generate the detailed sub-tasks. This ensures the high-level plan aligns with user expectations before diving into details.

## Target Audience

Assume the primary reader of the task list is a **junior developer** who will implement the feature. Tasks should be clear, specific, and actionable.

## Usage with Claude Code

```bash
# Basic usage
claude-code "Please read /tasks/prd-[feature-name].md and use workflows/generate-tasks.md to create a task list. First show me the high-level parent tasks, wait for my confirmation ('Go'), then generate detailed sub-tasks. Save to /tasks/tasks-prd-[feature-name].md"

# Example
claude-code "Please read /tasks/prd-user-profile.md and use workflows/generate-tasks.md to create a task list. First show me the high-level parent tasks, wait for my confirmation ('Go'), then generate detailed sub-tasks. Save to /tasks/tasks-prd-user-profile.md"
```

## Important Notes

- **Two-phase approach**: Parent tasks first, then sub-tasks after confirmation
- **Wait for "Go"**: Do not proceed to sub-tasks without user confirmation
- **File naming**: Output file should be `tasks-[prd-filename].md`
- **Junior developer friendly**: Keep all tasks clear and actionable