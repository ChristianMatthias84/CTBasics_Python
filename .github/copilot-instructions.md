# GitHub Copilot Instructions

## Core principle

You are working on an existing CT framework.

Your primary goal is to keep the project clean, maintainable, understandable, structurally consistent, and free from redundant implementations.

Before writing new code, inspect the existing codebase and determine whether the required functionality already exists or can be extended.

## 1. Search before implementing

Before creating any new function, method, helper, class, or module, search for existing functionality and related names first.

## 2. Avoid redundant functionality

Do not create duplicate helper functions or alternative implementations of the same behavior.

## 3. Find the proper location

Prefer extending existing classes or modules before creating new helpers or classes.

## 4. Comments and documentation

Add comments where algorithmic assumptions or numerical behavior are non-obvious.

## 5. Python-porting rules

- Keep the repository layout aligned with the MATLAB project structure.
- Provide MATLAB-style wrapper modules when compatibility matters.
- Keep raw-data and external toolbox dependencies deferred unless explicitly requested.
- Validate through focused pytest runs and document the checks.
