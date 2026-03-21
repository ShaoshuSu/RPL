#!/usr/bin/env python3
"""Policy runner with loco-manip depth support.

Applies runtime patches then delegates to the upstream holosoma_inference
run_policy entry point — same pattern as holosoma_rpl.run_sim.
"""

from holosoma_rpl_inference.patches import apply_runtime_patches

apply_runtime_patches()

from holosoma_inference.run_policy import main  # noqa: E402

if __name__ == "__main__":
    main()
