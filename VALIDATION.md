# Validation record

Validated in the creation environment:

- Python source compilation for both projects.
- `mapf-transformer-policy`: 7/7 tests passed.
- `pogema-mapf-transformer`: 3/3 tests passed.
- Synthetic trajectory generation.
- Two optimizer steps with the tiny configuration.
- Checkpoint save/load and offline action inference.
- Editable installation and CLI entry-point discovery.

The policy and companion core were tested without importing POGEMA. The current
POGEMA GitHub API was checked against its source, but a live simulator run was
not possible in this container because the optional upstream package was not
available for the container's Python/runtime combination. The adapter therefore
uses lazy imports and explicit API validation, and the POGEMA-dependent commands
fail with a targeted installation message when the optional dependency is
missing.
