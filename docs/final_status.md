# Final project status

This Python project covers the complete non-raw-data CTBasics functionality that is currently available in the MATLAB framework.

Included:
- descriptors and scanner geometry model
- phantom generation and motion/spectral logic
- forward projection and reconstruction helpers
- preprocessing and MAR-style helpers
- visualization and project-level utilities

Deferred intentionally:
- Siemens external raw-data readers and raw-data parsing that depend on external toolbox data and scanner-specific file formats.

The package intentionally keeps a single Python code tree under `src/ctbasics` and avoids duplicate wrappers or mirrored source trees.
