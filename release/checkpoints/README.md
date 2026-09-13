# CrossPath checkpoint bundle

This local release bundle contains the lightweight modules preserved after the
CIRR experiments:

- `cirr_target_refinement_e29.pt`: the selected target-refinement checkpoint
  used for the best CIRR validation result in the paper.
- `cirr_router_e30.pt`: the E30 retrieval-feature query router.
- `cirr_procrustes_e31.pt`: the E31 orthogonal compatibility map.
- `cirr_candidate_compatibility_group{1,4}_e32.pt`: the E32 candidate heads.

`CHECKSUMS.sha256` records SHA-256 hashes for every binary. The binaries and
the upload archive are ignored by Git; this README and checksum manifest are
safe to publish. The large frozen MCoT/DQU/FashionMV endpoint checkpoints are
not duplicated here and remain subject to their upstream licenses.

Local upload archive:

- `CrossPath_CIRR_lightweight_checkpoints_20260913.zip`
- SHA-256: `09256a368411fa6252b8d5c49b206ef33f29c855e4e292e646946dd1ccc07998`

Two selected FashionGen router checkpoints still need to be copied from the
experiment server when it is reachable:

- `/root/autodl-tmp/weave/runs/CrossPath_FashionGen_JointMatrix_20260820_v1/internal/gate/gate_width128.pt`
- `/root/autodl-tmp/weave/runs/CrossPath_A1seedpair_20260816_internal_v1/gate/gate_width128.pt`
