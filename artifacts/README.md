# Artifact policy

The active tree keeps only the current locked protocol, compact outcomes, actual PBS accounting,
and references needed to reproduce the final experiment. Large rollout arrays, videos, datasets,
and checkpoints live in ignored SSD or NSCC paths while active.

Historical campaign ledgers do not remain alongside current scientific inputs. They are recoverable
from the immutable tag documented in [`LEGACY_CAMPAIGNS.md`](LEGACY_CAMPAIGNS.md). A new campaign
gets one compact machine-readable manifest and one outcome bundle; per-job metadata is folded into
that bundle rather than accumulated as hundreds of standalone files.
