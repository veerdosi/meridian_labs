# Artifact policy

Large and regenerable data lives under this ignored directory on the external SSD. Only small
provenance manifests in `artifacts/manifests/` are versioned. Each manifest records source,
revision, checksum when available, storage URI, creation command, and related experiment/job IDs.

Storage retention keeps compact JSONL outcomes, logs, hashes, manifests, and the current selected
checkpoint. Raw `trajectory.npz` arrays and superseded checkpoint copies may be pruned after their
outcomes and provenance are versioned because they are reproducible from the locked plans and
checkpoint. Manifests record any removed checkpoint backup explicitly.
