# Checkpoint policy

Model weights are not stored in ordinary Git. Each canonical checkpoint must have one row in `manifest.tsv` containing its paper role, original location, size, SHA-256 digest, configuration, and eventual external download location.

Do not add sweep checkpoints or entire run directories. Only the exact checkpoint used for the final reported evaluation should be retained externally.

