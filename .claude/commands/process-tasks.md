---
name: Process Task List
description: Execute tasks one at a time with user approval
version: 1.0
tags: [execution, implementation, workflow]
---

# Task List Processing Workflow

## Goal
To systematically execute tasks from a task list file, working on ONE sub-task at a time with user approval between each task. This ensures quality control and allows for verification at each step.

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

## Core Rules

### ⚠️ Critical Rules - MUST Follow
1. **One sub-task at a time**: Do **NOT** start the next sub-task until you ask the user for permission and they say "yes" or "y"
2. **Immediate completion**: When you finish a sub-task, immediately mark it as completed by changing `[ ]` to `[x]`
3. **Parent task completion**: If **ALL** subtasks underneath a parent task are now `[x]`, also mark the **parent task** as completed
4. **Stop and wait**: Stop after each sub-task and wait for the user's go-ahead

## Process

### Step 1: Read Task List
1. Read the specified task list file
2. Identify the first uncompleted sub-task
3. Understand the context and requirements

### Step 2: Execute Sub-Task
1. Work on the current sub-task completely
2. Implement all necessary code, files, or changes
3. Ensure the sub-task is fully completed

### Step 3: Update Task List
1. Open the task list file
2. Mark the completed sub-task as `[x]`
3. If all sub-tasks under a parent are complete, mark the parent as `[x]`
4. Save the updated task list file

### Step 4: Ask for Permission
1. Inform the user that the sub-task is complete
2. Ask: **"Task [task-id] is complete. Ready to proceed to the next task? (yes/y to continue)"**
3. Wait for user response
4. Only proceed if user responds with "yes", "y", or similar affirmative

### Step 5: Continue or Stop
- If user approves: Move to the next uncompleted sub-task
- If user provides feedback: Address the feedback before proceeding
- If all tasks complete: Celebrate completion!

## Task List Maintenance

### Update as You Work
- Mark tasks and subtasks as completed (`[x]`) per the protocol above
- Add new tasks as they emerge during implementation
- Keep the task list file current and accurate

### Maintain "Relevant Files" Section
- List every file created or modified
- Give each file a one-line description of its purpose
- Update this section as you work

## Example Interaction

```
AI: Starting task 1.1: Create user profile component

[AI implements the component]

AI: ✅ Completed: Create user profile component
    Task 1.1 is complete. Ready to proceed to the next task? (yes/y to continue)

User: yes

AI: Starting task 1.2: Add profile validation logic

[AI implements validation]

AI: ✅ Completed: Add profile validation logic
    Task 1.2 is complete. Ready to proceed to the next task? (yes/y to continue)
```

## Completion Protocol

### Sub-Task Completion
```markdown
- [x] 1.1 Create user profile component (COMPLETED)
```

### Parent Task Completion
```markdown
- [x] 1.0 Create User Profile Feature (ALL SUB-TASKS COMPLETE)
  - [x] 1.1 Create user profile component
  - [x] 1.2 Add profile validation logic
  - [x] 1.3 Add profile save functionality
```

## Status Messages

### Task Complete
```
✅ Completed: [task description]
Task [task-id] is complete. Ready to proceed to the next task? (yes/y to continue)
```

### All Tasks Complete
```
🎉 All tasks completed successfully!
Feature implementation is complete. All tasks have been marked as finished.
```

## Usage with Claude Code

```bash
# Basic usage
claude-code "Please read /tasks/tasks-prd-[feature-name].md and use workflows/process-tasks.md to start implementing. Work on ONE sub-task at a time, mark it complete when done, and ask for my permission before moving to the next sub-task."

# Example
claude-code "Please read /tasks/tasks-prd-user-profile.md and use workflows/process-tasks.md to start implementing. Work on ONE sub-task at a time, mark it complete when done, and ask for my permission before moving to the next sub-task."
```

## Important Reminders

- **NEVER skip asking for permission** between sub-tasks
- **ALWAYS update the task list file** after completing each sub-task
- **WAIT for user confirmation** before proceeding
- **Keep tasks organized** and maintain the file structure
- **Add new tasks** if you discover additional work needed