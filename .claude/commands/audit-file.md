---
name: Audit File
description: Analyze and clean up a Python file to improve code quality and readability
version: 1.0
tags: [code-quality, refactoring, cleanup]
---

# Audit and Clean Python File

## Goal
To analyze a Python file for code quality issues and automatically clean it up, making it more concise, readable, and maintainable. Target: keep files under 500 lines when possible.

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

### Step 1: Initial Analysis
1. Read the specified Python file
2. Analyze current file statistics (lines, functions, complexity)
3. Identify all issues across the audit criteria

### Step 2: Issue Identification
Systematically check for these issues:

**Code Structure & Organization:**
1. **Unused functions** - Functions that are never called
2. **Legacy functions** - Deprecated or outdated methods
3. **Redundant code** - Duplicate or unnecessary code blocks
4. **Long functions** - Functions >50 lines that should be broken down
5. **Duplicate code blocks** - Similar patterns that can be extracted
6. **Hard-coded values** - Magic numbers that should be constants
7. **Poor variable scope** - Variables with unnecessarily broad scope

**Efficiency & Best Practices:**
8. **Inefficient code** - Suboptimal algorithms or data structures
9. **Non-recommended methods** - Long numpy code when pandas one-liners exist
10. **Over-nesting** - More than 3 nested levels
11. **Memory inefficiency** - Unnecessary data copying, large intermediate variables

**Scientific Code Quality:**
12. **Missing input validation** - No checks for data dimensions, types, ranges
13. **Unclear scientific parameters** - Unexplained thresholds, cutoffs, constants
14. **Non-reproducible elements** - Missing random seeds, unstable operations
15. **Single-responsibility violations** - Functions doing too many things

**Python Standards:**
16. **Missing type hints** - Functions without proper typing
17. **Inconsistent naming** - Variables not following snake_case, unclear abbreviations
18. **Import issues** - Unused imports, non-standard import patterns
19. **Exception handling** - Generic try/except blocks, missing error context

**Documentation & Readability:**
20. **Missing docstrings** - Functions without proper documentation
21. **Unclear comments** - Outdated or confusing comments
22. **Complex expressions** - Multi-line expressions that could be simplified

### Step 3: Present Summary
Create a concise summary of major issues found:

```
## Audit Summary for [filename]

**Current Stats:**
- Lines: [current] → Target: <500
- Functions: [count]
- Max nesting depth: [level]

**Major Issues Found:**
- [Issue category]: [count] instances
- [Issue category]: [count] instances
- [Brief description of most significant problems]

**Proposed Changes:**
- Remove [X] unused functions
- Break down [X] long functions
- Simplify [X] complex expressions
- Add [X] missing type hints
- [Other major improvements]

**Estimated line reduction: [X] lines**

Ready to proceed with cleanup? (yes/y to continue)
```

### Step 4: Wait for User Approval
- Present the summary to the user
- Ask: **"Ready to proceed with cleanup? (yes/y to continue)"**
- Wait for user confirmation before making any changes

### Step 5: Apply Fixes
1. Make all approved changes to the original file
2. Ensure changes follow the development guidelines
3. Preserve all functional behavior
4. Maintain scientific accuracy and reproducibility

### Step 6: Verification
1. Check that the file still runs without errors
2. Verify that key functions still work as expected
3. Confirm line count reduction if applicable

## Usage with Claude Code

```bash
# Basic usage
claude-code "Using .claude/commands/audit-file.md, analyze and clean up [file-path]. Show me the audit summary first and wait for my approval before making changes."

# Example
claude-code "Using .claude/commands/audit-file.md, analyze and clean up src/eeg_processor/pipeline.py. Show me the audit summary first and wait for my approval before making changes."
```

## Important Notes

- **Original file modification**: Changes are made directly to the original file
- **Use git for backups**: Rely on version control for change tracking
- **User approval required**: Always wait for confirmation before applying fixes
- **Preserve functionality**: Never change the external behavior of functions
- **Scientific accuracy**: Maintain all scientific logic and parameters
- **Target length**: Aim for <500 lines when possible without sacrificing clarity