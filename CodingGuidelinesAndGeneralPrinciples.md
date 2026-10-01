# Coding Guidelines and General Principles

## Core principle

We are working on an existing CT framework and must maintain it in a clean, maintainable, and understandable way.

Before creating new code, inspect the existing implementation and determine whether the required functionality already exists or can be extended.

## 1. Search before implementing

Before creating a new function, method, helper, class, or module:

1. Search for existing functionality with the same purpose.
2. Search for similarly named functions and methods.
3. Search for related behavior under a different name.
4. Identify the logical owner of the functionality.
5. Check whether an existing implementation can be extended instead of creating a new one.

Avoid creating a new implementation without checking whether a similar one already exists.

## 2. Avoid redundant functionality

The framework should have one clear implementation for each piece of functionality whenever reasonably possible.

Do not add duplicate helpers, copy-paste logic, or near-identical algorithms.

## 3. Helpers need a real benefit

A helper should only be introduced when it adds clear architectural or readability value.

Before creating a helper, check whether the logic belongs in an existing module, class, or utility collection.

## 4. Find the correct architectural location

For new functions and helpers, determine the best home before implementing them.

Prefer this order:

1. Extend an existing function.
2. Extend an existing class.
3. Extend an existing utility or function collection.
4. Add it to an existing module.
5. Create a helper only if necessary.
6. Create a new class only if clearly justified.

## 5. Comments and documentation

Code should be well commented and explain the reasoning behind algorithmic choices, assumptions, geometry conventions, performance decisions, and edge cases.

## 6. Python porting rules

- Preserve the original domain structure and naming where practical.
- Keep MATLAB-style entry-point names available via thin wrappers.
- Prefer Pythonic module naming internally while mirroring the MATLAB layout in package-level wrappers.
- Keep external raw-data readers deferred unless the task explicitly calls for them.
