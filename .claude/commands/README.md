# AI Development Workflow Templates

This directory contains **Markdown workflow templates** for structured AI-assisted development using Claude Code.

## Available Workflows

### 1. **Create PRD** (`create-prd.md`)
Generate a comprehensive Product Requirements Document for a new feature.

**Usage:**
```bash
claude-code "Using workflows/create-prd.md, help me create a PRD for [feature description]"
```

### 2. **Generate Tasks** (`generate-tasks.md`)
Convert a PRD into a detailed, actionable task list with parent tasks and sub-tasks.

**Usage:**
```bash
claude-code "Using workflows/generate-tasks.md, read /tasks/prd-[feature].md and create a task list. Show parent tasks first, wait for 'Go', then generate sub-tasks."
```

### 3. **Process Tasks** (`process-tasks.md`)
Execute tasks one at a time with user approval between each step.

**Usage:**
```bash
claude-code "Using workflows/process-tasks.md, read /tasks/tasks-prd-[feature].md and start implementing. Work on ONE sub-task at a time."
```

## Complete Workflow Example

```bash
# Step 1: Create PRD
claude-code "Using workflows/create-prd.md, help me create a PRD for user authentication system"

# Step 2: Generate tasks (after PRD is created)
claude-code "Using workflows/generate-tasks.md, read /tasks/prd-user-auth.md and create a task list. Show parent tasks first, wait for 'Go', then generate sub-tasks."

# Step 3: Execute tasks (after task list is created)
claude-code "Using workflows/process-tasks.md, read /tasks/tasks-prd-user-auth.md and start implementing. Work on ONE sub-task at a time."
```

## Why Markdown?

We chose **Markdown** as the single format because it:

- ✅ **Human-readable** - Easy to read and understand
- ✅ **Version control friendly** - Works great with Git
- ✅ **Widely supported** - Every text editor supports it
- ✅ **Simple yet structured** - YAML frontmatter + readable content
- ✅ **Easy to modify** - No complex syntax or schemas

## File Structure

Each workflow file contains:

```markdown
---
name: Workflow Name
description: Brief description
version: 1.0
tags: [tag1, tag2]
---

# Workflow Title

## Goal
What this workflow accomplishes

## Process
Step-by-step instructions

## Usage with Claude Code
Command examples
```

## Key Workflow Features

- **Structured approach** - Clear phases from planning to implementation
- **Human-in-the-loop** - User approval required between major steps
- **Junior developer friendly** - Clear, explicit instructions
- **Progress tracking** - Checkbox-based task completion
- **File organization** - All artifacts saved to `/tasks/` directory

## Getting Started

1. Choose a workflow file based on your current phase
2. Use the provided Claude Code command examples
3. Follow the step-by-step process in each workflow
4. Review and approve work at each checkpoint

Happy AI-assisted developing! 🚀