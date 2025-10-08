---
name: Create PRD
description: Generate a Product Requirements Document for a new feature
version: 1.0
tags: [planning, documentation, requirements]
---

# Create Product Requirements Document (PRD)

## Goal
To create a detailed Product Requirements Document (PRD) in Markdown format, based on an initial user prompt. The PRD should be clear, actionable, and suitable for a junior developer to understand and implement the feature.

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

### Step 1: Clarifying Questions
Before writing the PRD, ask clarifying questions to gather sufficient detail. Focus on understanding the "what" and "why" of the feature.

**Questions to ask:**
- **Problem/Goal:** What problem does this feature solve for the user? What is the main goal?
- **Target User:** Who is the primary user of this feature?
- **Core Functionality:** What are the key actions a user should be able to perform?
- **User Stories:** Can you provide user stories? (As a [user], I want to [action] so that [benefit])
- **Acceptance Criteria:** How will we know when this feature is successfully implemented?
- **Scope/Boundaries:** Are there specific things this feature should NOT do?
- **Data Requirements:** What kind of data does this feature need?
- **Design/UI:** Are there any design mockups or UI guidelines to follow?
- **Edge Cases:** What potential edge cases or error conditions should we consider?

### Step 2: Generate PRD
Based on the answers, create a PRD with these sections:

1. **Introduction/Overview** - Brief description and problem it solves
2. **Goals** - Specific, measurable objectives
3. **User Stories** - User narratives describing feature usage
4. **Functional Requirements** - Numbered list of specific functionalities
5. **Non-Goals (Out of Scope)** - What this feature will NOT include
6. **Design Considerations** (Optional) - UI/UX requirements
7. **Technical Considerations** (Optional) - Known constraints or dependencies
8. **Success Metrics** - How success will be measured
9. **Open Questions** - Remaining questions needing clarification

### Step 3: Save PRD
Save the document as `/tasks/prd-[feature-name].md`

## Template

```markdown
# PRD: [Feature Name]

## Introduction/Overview
[Brief description of the feature and the problem it solves]

## Goals
- [Specific objective 1]
- [Specific objective 2]

## User Stories
- As a [type of user], I want to [perform action] so that [benefit]
- As a [type of user], I want to [perform action] so that [benefit]

## Functional Requirements
1. The system must [specific functionality]
2. The system must [specific functionality]
3. Users should be able to [specific action]

## Non-Goals (Out of Scope)
- This feature will NOT [excluded functionality]
- We are NOT implementing [excluded feature]

## Design Considerations
[Any UI/UX requirements or mockup references]

## Technical Considerations
[Any known technical constraints or dependencies]

## Success Metrics
- [Measurable success criterion]
- [Measurable success criterion]

## Open Questions
- [Question requiring further clarification]
- [Question requiring further clarification]
```

## Usage with Claude Code

```bash
claude-code "Using the workflow in workflows/create-prd.md, help me create a PRD for [feature description]"
```