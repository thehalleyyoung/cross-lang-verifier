# Cross-architecture replay

> Claim **C65-cross-architecture-replay** · module `src/ub_oracle/arch_replay.py`

Cross-architecture replay checks whether a confirmed witness has the same
semantic verdict on each CPU architecture whose real runner is available. The
report records the native host architecture, the exact architectures requested
(`arm64`, `x86_64` by default), one verdict-layer hash per available architecture,
and an explicit unavailable reason for every architecture without a native or
emulated runner.

The implementation is deliberately conservative: it only runs through the real
`ReexecHarness` on a genuinely available architecture. The detector for
architecture-dependent verdict changes is unit-tested with synthetic fixtures,
but those reports are marked `synthetic=true` and are not empirical evidence.

```bash
make cross-arch-check
PYTHONPATH=src python -m ub_oracle.arch_replay
```

On a single-ISA developer machine this usually produces one native replay and
one unavailable row. That is an honest negative/partial result, not a fabricated
claim that both ISAs were executed.
