"""Varied natural operational prompt templates (>=16 per family) for Agent Lens Insights clustering."""

from __future__ import annotations

LEDGER_OPENERS: tuple[str, ...] = (
    "Audit today's merchant settlement batches for account `{merchant}`.",
    "Run a full close-of-day settlement reconciliation on merchant `{merchant}`.",
    "We have an end-of-day payout discrepancy check queued for `{merchant}`.",
    "Verify settled USD cents, rollover cents, and chargeback reserve for `{merchant}`.",
    "Please reconcile all four settlement batches and adjustment chains for `{merchant}`.",
    "Close out the daily multi-currency ledger for `{merchant}` using the live policy rules.",
    "Finance flagged `{merchant}` for manual sign-off before wire release; reconcile its batches.",
    "Compute the exact post-fee USD settlement and next-day rollover for `{merchant}`.",
    "Inspect every batch, adjustment record, and rounding rule for `{merchant}` and file the totals.",
    "Perform a strict cent-exact payout audit across `{merchant}`'s settlement batches.",
    "Before treasury sweeps `{merchant}`, verify active adjustments, net settled cents, and reserve.",
    "Check `{merchant}`'s multi-currency settlement ledger and submit the verified reconciliation.",
    "Walk `{merchant}`'s batch entries and superseding adjustments and record the final numbers.",
    "Reconcile `{merchant}` against the current cutoff, fee tiers, and FX policy.",
    "Audit `{merchant}`'s settlement run from raw batch entries through active adjustments.",
    "Prepare the authoritative settlement figures for `{merchant}` by querying all ledger tools.",
)

FAULT_TURN1_OPENERS: tuple[str, ...] = (
    "Page alert on cluster `{cluster}`: identify the earliest true upstream root-cause service.",
    "Incident `{incident_id}` just fired on `{cluster}`; trace the telemetry and isolate the root cause.",
    "We are seeing cascading latency across `{cluster}` (`{incident_id}`). Find where the breach started.",
    "Triage `{incident_id}` in `{cluster}` by checking topology, maintenance windows, and both telemetry pages.",
    "On-call handoff for `{cluster}` (`{incident_id}`): determine which service breached first outside maintenance.",
    "Investigate `{incident_id}` on `{cluster}` and separate downstream cascade noise from the root service.",
    "Latency SLOs tripped across `{cluster}` under `{incident_id}`; find the earliest non-maintenance root breach.",
    "Pull shard telemetry for `{cluster}` (`{incident_id}`), failing over if primary times out, and name the root cause.",
    "Diagnose `{incident_id}` in `{cluster}` using the dependency graph and full 12-tick telemetry window.",
    "Pinpoint the originating service and breach tick for `{incident_id}` on `{cluster}`.",
    "Run root-cause attribution on `{cluster}` incident `{incident_id}`, accounting for GC pauses and maintenance.",
    "Check `{cluster}` telemetry for `{incident_id}` across all pages and submit the verified root cause.",
    "Separate scheduled maintenance spikes from real SLO breaches on `{cluster}` (`{incident_id}`).",
    "Audit telemetry ticks 1..12 for `{cluster}` (`{incident_id}`) and file the upstream root-cause report.",
    "Which service actually initiated `{incident_id}` on `{cluster}` once upstream cascades are ruled out?",
    "Trace `{incident_id}` across `{cluster}` replicas and report the earliest unmasked root-cause service.",
)

SPEC_TURN1_OPENERS: tuple[str, ...] = (
    "Synthesize a 3-wave deployment schedule for release `{release_id}` that satisfies every constraint.",
    "Build and commit a valid three-wave rollout plan for `{release_id}`.",
    "Release engineering needs a verified 3-wave rollout order for `{release_id}`.",
    "Arrange the seven services in `{release_id}` into three waves without violating memory or ordering rules.",
    "Plan the `{release_id}` production rollout across waves 1..3, compute its digest, and commit Turn 1.",
    "Construct a compliant 3-wave deployment plan and canary seal for `{release_id}`.",
    "Solve the wave assignment and ordering constraints for `{release_id}` and commit the verified plan.",
    "Generate Turn 1's deployment waves for `{release_id}`, checking peak concurrent memory carefully.",
    "Schedule all seven `{release_id}` services across three waves, ground the digest via tool, and commit.",
    "Prepare a conflict-free 3-wave rollout schedule for `{release_id}` with exact seal formatting.",
    "Find a feasible 3-wave service ordering for `{release_id}` under the spike memory cap.",
    "Assemble the `{release_id}` wave plan, verify schema and checksum with tools, and submit Turn 1.",
    "Produce a valid 3-wave release topology for `{release_id}` and record its signed digest.",
    "Map `{release_id}`'s services into waves 1, 2, and 3 while respecting hard dependencies and memory.",
    "Draft, checksum, and commit the initial three-wave rollout plan for `{release_id}`.",
    "Compute a valid 3-wave deployment sequence and transformed canary seal for `{release_id}`.",
)

PATCH_OPENERS: tuple[str, ...] = (
    "Module `{module_name}` is failing edge-case invariants in production; inspect, test, and submit a clean patch.",
    "Read `{module_name}`, find every discrepancy against its docstring specification, and submit a fixed module.",
    "Audit `{module_name}` carefully (smoke tests only cover basic paths) and patch all specification bugs.",
    "Fix the implementation defects in `{module_name}` so every docstring invariant holds, then submit the patch.",
    "Inspect `{module_name}`, verify behavior with `run_smoke_tests`, and fix all subtle boundary bugs.",
    "We suspect multiple off-by-one or state bugs in `{module_name}`; review the spec and submit a corrected file.",
    "Repair `{module_name}` to match its documented contract exactly without changing public signatures.",
    "Review `{module_name}` line by line against its docstring rules, test locally, and submit your patch.",
    "Track down the semantic bugs hiding in `{module_name}` beyond the smoke suite and submit a complete fix.",
    "Bring `{module_name}` into strict compliance with every edge-case rule in its docstring.",
    "Debug `{module_name}` using `read_source` and `run_smoke_tests`, then submit the repaired Python source.",
    "Patch `{module_name}` so both basic smoke checks and adversarial boundary invariants pass.",
    "Examine `{module_name}`'s state transitions and tie-breakers, fix any spec violations, and submit.",
    "Hardening pass on `{module_name}`: eliminate every bug relative to the docstring contract.",
    "Locate all planted logic errors in `{module_name}` and submit a self-contained corrected module.",
    "Verify and repair `{module_name}` against every invariant stated in its class docstring.",
)
