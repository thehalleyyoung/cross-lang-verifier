# SemRec / XLEV

`cross-lang-verifier` ships two Python CLIs for **C↔Rust source-level
equivalence analysis**:

- `semrec` for bounded, focused equivalence checking and benchmark runs
- `xlev` for project discovery, verification, fuzzing, and analysis workflows

The strongest verified claim for this repository is that it contains a real
SMT-backed analysis stack for comparing C and Rust functions, with explicit
front ends for source strings, source files, benchmark suites, and project
discovery. Any broader benchmark numbers should be treated cautiously unless you
rerun them locally.

## Install

```bash
python3 -m pip install -e .
```

## CLI entry points

```bash
python3 -m src.semrec_cli --help
python3 -m src.cli.main --help
```

The module entry points above correspond to:

- `semrec`: `verify`, `cegar`, `bench`
- `xlev`: `verify`, `discover`, `fuzz`, `analyze`, `benchmark`

## Correct usage examples

```bash
# Inline C / Rust comparison
python3 -m src.semrec_cli verify \
  --c-code 'int f(int x){return x+1;}' \
  --rs-code 'pub fn f(x:i32)->i32{x.wrapping_add(1)}' \
  --format text

# File-based project-level verification
python3 -m src.cli.main verify \
  --c-source examples/sample_project/c_impl/math_utils.c \
  --rust-source examples/sample_project/src/lib.rs \
  --format text

# Auto-discover candidate pairs in a mixed project
python3 -m src.cli.main discover \
  --cargo-dir examples/sample_project \
  --c-dir examples/sample_project/c_impl
```

## What the repository demonstrably contains

- SMT-backed equivalence checking in `src/`
- a growing catalogue of **undefined-behavior divergence oracles** — signed
  overflow, shift-out-of-range, division-by-zero, `INT_MIN/-1`, array
  out-of-bounds, strict aliasing, floating-point contraction, and an
  **uninitialized-read / definedness oracle** (`oracles/uninit_read.py`) built on
  a real three-point definedness-lattice dataflow analysis that flags reads of
  slots never written on all paths and confirms them across C→Rust/Go/Swift
  (same C source diverging at `-O0` vs `-O2` while the zero/default-initialized
  translation is defined and deterministic)
- source-string and source-file verification flows
- project discovery and benchmark runners in `src/cli/`
- a faithful counterexample minimizer (`src/ub_oracle/minimizer.py`) that shrinks
  each confirmed witness to a locally-minimal canonical form while preserving the
  exact UBSan diagnostic category — see the `make cex-quality` study
- a severity-ranked triage view and linter-style baseline/suppression files
  (`cross-lang-verify --triage` / `--write-baseline` / `--suppress`) so teams can
  adopt the checker on a large existing migration and fail CI only on *new*
  divergences
- an incremental cache (`--cache`) that re-verifies only changed units — keyed by
  unit content **and** the real toolchain version, so a compiler upgrade
  invalidates stale verdicts (~25× speedup at full reuse on the sample manifest)
- a self-contained, offline migration-risk HTML dashboard (`--dashboard`)
- a performance/scalability study (`make perf`) that times the *real* symbolic
  searches across every class and language pair and characterises the SMT
  backbone's growth out to 512-bit widths — the deterministic grid
  (`experiments/perf_curves/grid.json`) is checked for byte-identical
  regeneration in CI (`make perf-check`)
- a documented plugin **SDK** ([`docs/SDK.md`](docs/SDK.md)) with a worked
  *external* oracle ([`examples/plugins/float_cast_overflow_oracle.py`](examples/plugins/float_cast_overflow_oracle.py))
  proving a third party can add a brand-new divergence class — confirmed against
  real `clang`+`rustc` — without forking the engine
- an internal **red-team** (`make redteam`) that, for every oracle on every
  supported language pair, throws a battery of semantics-preserving adversarial
  mutations of a genuinely-divergent unit at the verifier and proves it never
  falsely returns "no divergence" — 63/63 cases confirmed divergent against real
  compilers, **zero soundness breaches** (the byte-reproducible adversarial grid
  is CI-checked via `make redteam-check`)
- a **branch-coverage ratchet** over the toolchain-independent brain of the tool
  (`make coverage` / `make coverage-check`): 20 curated core modules — the IR,
  oracle SPI, decision layer, symbolic searches, and finding pipeline — held at
  **91 %+ mean branch coverage** with a committed floor in `coverage_floor.json`
  that CI never lets regress
- an **interval-domain abstract-interpretation pre-pass** (`abstract_interp.py`)
  that, before any SMT call, *proves* a divergence class's undefined-behavior
  region unreachable under a unit's declared operating range and discharges it
  without invoking the solver — a sound accelerator (it only ever discharges
  no-divergence, never asserts one) that also lets a unit declare a safe range
  so it isn't flagged for a divergence that range forbids; the same range is
  honored by the oracles' searches so the fast path and the SMT path always agree
- a **counterexample-guided abstraction-refinement (CEGAR) loop** (`cegar.py`)
  for the *path-sensitive* fragments the interval pre-pass cannot discharge: it
  starts from the bare UB condition, and on every spurious model **learns one
  path-condition (`assume`) at a time**, terminating either in a proof that no
  guarded input is undefined or in a concrete witness — whose verdict is checked
  against exact enumeration of the UB region and whose witnesses are confirmed
  against the real C `-O0`/`-O2` builds
- a **k-induction engine for loops beyond bounded unrolling** (`kinduction.py`)
  so a no-divergence claim is no longer "we unrolled N times and saw nothing":
  the inductive step (optionally strengthened with auxiliary invariants the
  engine first *proves* inductive) certifies safety for an **unbounded** trip
  count, while the base case returns a reachable overflow witness at the **exact**
  iteration the loop goes undefined — confirmed against real compilers (the C
  loop traps under UBSan at precisely that trip count, and the SAFE verdict
  transfers across C→Rust/Go/Swift)
- an **ABI / interop-divergence checker** (`abi_layout.py`) for the FFI boundary
  itself, covering **structs, unions, enums and nested aggregates**: it computes
  the *exact* C-ABI layout of a shared type (size, align, per-field offsets —
  confirmed field-by-field against real `clang` `offsetof`/`_Alignof`) and flags
  an interop hazard **iff** a padding-optimizing representation would reorder a
  struct, **or iff** a C `enum` and a default-repr Rust *fieldless* enum of the
  same arity have different widths (C = `int`/4 bytes vs Rust's auto-sized 1–4
  bytes). Real `rustc` confirms `#[repr(C)]` mirrors the C layout/width exactly
  while the default `repr(Rust)` genuinely diverges **precisely when** predicted,
  and Go's declaration-order layout matches C — so a layout/width hazard (a silent
  memory-safety bug no value-level oracle can see) is never fabricated
- a **byte-addressed provenance memory model** (`memory_model.py`) that decides
  spatial (out-of-bounds) and temporal (use-after-free, double-free) safety on
  whole allocation/free/load/store **traces**: every pointer carries the
  *provenance* of its allocation, so an access is legal **iff** it stays inside
  that object's bytes (never an adjacent one) and only while the object is alive.
  Each trace is emitted as C, compiled under **AddressSanitizer**, and ASan is
  checked to trap **iff** the model predicts a fault — reporting the *same* fault
  kind — so the model never invents (or misses) a memory bug on the traces it
  generates
- a **pointer-provenance (PNVI) model** (`provenance.py`) for the part of the C
  memory model where translation bugs actually live: every pointer carries the
  *provenance* of one allocation, a one-past-the-end pointer is **formable and
  comparable but not dereferenceable**, pointer arithmetic preserves provenance,
  and an integer round-trip recovers provenance **only** via exposure (PNVI-ae).
  The confirmable distinctions — forming vs dereferencing one-past-the-end, and
  in-bounds arithmetic round-trips — are checked against **AddressSanitizer** on
  real code, and the general provenance interface is documented for other pairs
- an **ownership / borrow-fact extractor** (`ownership.py`) that takes the Rust
  borrow checker's verdict as ground truth: the idiomatic *safe* translation of a
  mutably-aliasing C idiom is **rejected** by real `rustc` (`E0499`/`E0502`/
  `E0382`), while disjoint borrows are accepted and the `unsafe` raw-pointer
  re-expression compiles — exactly the translator's dilemma. Each accept/reject
  verdict (and error code) is *observed* by compiling, not assumed, and the
  interface is documented to retarget to other safety models
- a **structural function aligner** (`unit_alignment.py`) that matches each C
  function to its translated counterpart by **signature compatibility** (arity +
  C→target type families, with an arity mismatch acting as a hard veto) and
  **self-reinforcing call-graph agreement**, using name similarity only as a
  tiebreak. On a renamed module engineered to be *adversarial* to name matching
  (a 2-arg `add` whose name is closest to a 1-arg `add_one`), it recovers the
  ground-truth pairing exactly while a name-only baseline does not — so divergence
  oracles compare the *right* function pairs even across heavy renaming
- a **soundness-frontier detector** (`foreign_effects.py`) that keeps the tool
  honest: the oracles reason about a *pure* C fragment, so this detector flags
  every construct that leaves it — `volatile` accesses, inline `__asm__`, calls
  to undefined `extern` functions (true FFI), atomics, `setjmp`/`longjmp`, signal
  handlers — and makes the tool **abstain loudly** (a named reason per site)
  instead of guessing, while ordinary libc-only code stays CLEAR. Each abstention
  is *justified against real clang IR*: a `volatile` loop keeps four separate
  `load volatile` where the pure version coalesces to one (so a value-folding
  model is provably unsound), inline asm is an opaque `call ... asm`, an `extern`
  callee is an undefined `declare`, and an atomic lowers to `load atomic`
- a **concurrency / data-race oracle** (`concurrency.py`) that turns the migration
  story into a *checked* fact: the same unsynchronized shared-counter idiom is
  confirmed to be a **data race on both the C source (ThreadSanitizer) and the Go
  target (`go run -race`)** — while Rust rejects it at compile time (Send/Sync) —
  and every synchronized variant (mutex, lock-free atomic, read-only sharing) runs
  clean under both detectors. A pattern is only ever called a race when a real
  sanitizer actually fires on a real binary
- an **indirect-call resolver** (`indirect_calls.py`) so real codebases that
  dispatch through function-pointer tables (syscall tables, plugin registries,
  hand-rolled vtables) don't lose their call-graph edges: the precise points-to
  set of `table[k](...)` is exactly the functions in the table's initializer, and
  it refines a conservative **signature-typed** set that excludes wrong-signature
  decoys. The resolution is **proven exact against real execution** — an
  instrumented build is compiled and run, and the functions actually reached
  through the table equal the predicted set (never escaping it)
- a **real-preprocessing front door** (`preprocess.py`) — analysis runs only after
  `clang -E`, because a C program's meaning is fixed only post-preprocessing. It
  proves three load-bearing facts on real binaries: an unparenthesized
  function-like macro is **semantically load-bearing** (`MUL(1+1,2)` → `1+1*2` → 3
  vs the parenthesized 4, and the hazardous macro is detected up front), `#ifdef`
  conditionals **select the program**, and `#include` symbols **resolve**
- a **behavior-accurate libc model** (`libc_model.py`): executable specs for the
  most-hit runtime surface (`strlen`, `strcmp`, `strncmp`, `memcmp`, `memcpy`,
  `memset`, `strchr`) proven against the **real libc** by randomized differential
  testing — each spec runs on the host libc through a compiled harness and matches
  the pure model on hundreds of inputs (zero mismatches), encoding the exact
  contract (sign-only comparisons, NUL preconditions, `memcpy` no-overlap). The
  `LibcSpec`/`confirm_spec` framework is runtime-agnostic and reusable
- **high-fidelity ingestion from the compilers' own IRs** (`ir_ingest.py`): rather
  than re-parsing source by hand, signatures and ownership facts are recovered from
  clang's `-ast-dump=json` (exact parameter names, types and storage classes —
  `static char *dup_first(const char *)`) and rustc's `--emit=mir` (a by-value
  non-`Copy` parameter such as `Vec<i32>` shows a `move`/`drop` of its local and is
  recorded as *consumed*; a `Copy` `i32` is not — ownership for free, from the
  compiler itself), with builtins filtered by source-location provenance and the
  ingesters self-confirming against the real clang/rustc on every run
- **whole-project ingestion** (`project_ingest.py`): scales from a single file to
  a whole build tree on both sides — the source side reads a Clang
  `compile_commands.json` database (the CMake/Bear standard), recovers each unit's
  `-I` directories and unions all translation units' symbols into one
  `ProjectModule`; the target side enumerates a Cargo workspace via `cargo
  metadata` (the build graph Cargo itself uses), discovering every member package,
  target and source root — both proven against the real clang/cargo on generated
  multi-file projects
- a **solver portfolio** (`solver_portfolio.py`): z3 (in-process) and boolector
  (out-of-process) race each SMT-LIB2 query in parallel under a shared timeout;
  the first decisive answer wins and every answering solver is cross-checked for
  agreement (a disagreement is a loud `UNKNOWN`, never a silently-chosen verdict).
  A robustness battery over divergence-relevant bit-vector classes is solved by
  every available solver and matched against ground truth
- a **differentially-fuzzed C frontend** (`frontend_fuzz.py`): a seeded generator
  emits random but always-compilable well-typed C, and every program is diffed
  against clang's ground truth — names, arity, return/parameter types and storage
  class must match exactly. Dozens of programs survive each fixed-seed run with
  zero parse-divergences and zero crashes, and malformed input returns `None`
  rather than raising
- a **per-language conformance suite** (`conformance.py`): a curated regression
  corpus pairing real C/Rust constructs with their exact expected lowerings —
  array-decay, function-pointer params, typedef/enum/struct-tag preservation and
  `static` linkage on the C side; by-value `Vec`/`Box`/`String` *moves* vs
  reference/`Copy` non-moves on the Rust side — checked against the real
  compilers so any drift in how a construct lowers turns a case red
- an **evaluation-order / sequencing oracle** (`eval_order.py`): unsequenced
  modification (`i = i++ + i++`, `g(i++, i++)`) is genuine C undefined behavior,
  detected precisely via clang `-Wunsequenced` and answered with a loud `ABSTAIN`
  (no target translation picking a concrete order can be proven equivalent to
  another legal C compilation); unspecified argument-evaluation order with side
  effects is documented as the sequencing soundness frontier
- a **known-bug / CVE corpus** (`cve_corpus.py`): a curated "we catch real bugs"
  table of UB-rooted weakness classes — division by zero (CWE-369), out-of-bounds
  read (CWE-125), signed overflow (CWE-190), oversized shift (CWE-758), INT_MIN/-1
  (CWE-682) — each translated to Rust *and* Go and **verified end-to-end** by
  compiling and running both sides: C traps under the sanitizer while the target
  is defined and deterministic. Every catch is executed, not asserted
- a **labeled ground-truth set** (`ground_truth.py`): **≥500** `(C program,
  target translation)` pairs across **two** language pairs whose `divergent`/
  `equivalent` label is fixed by *bounded enumeration + real sanitizers* (UBSan
  traps and observable-output comparison), not by the oracle — the independent
  substrate that makes a precision/recall claim defensible. On a full toolchain
  the sanitizer-established label agrees with the constructed label on every
  sampled item, across both languages and both labels
- a **scale-measurement harness** (`scale_measure.py`): drives the labeled corpus
  through the real decision procedure recording **time/memory/verdict/abstention
  per item** and emits a **canonical results JSON** whose *verdict layer* is
  content-hashed (sorted keys, stable separators) while timing/memory are kept in
  a separate, explicitly non-hashed section — so two runs on the same toolchain
  reproduce the identical `content_hash` even though wall-clock and RSS vary
- a **head-to-head vs existing tools** (`external_baselines.py`): probes the
  machine for every relevant tool category (bounded model checking /
  single-language equivalence, symbolic execution, translation validation, static
  analysis, verified transpilers) and shows — concretely and executed — that none
  ingests a cross-language `(C, target)` pair. The realizable single-language /
  translation-validation proxy (a same-language O0-vs-O2 differential) is run on
  real divergent items: on the provably-blind classes (div-by-zero, INT_MIN/-1) it
  finds **nothing** while the oracle catches **every** one — a total
  false-negative gap, with value-exploited UB reported honestly in the per-class
  breakdown
- an **external replication kit** (`Dockerfile` + `make reproduce-kit` +
  `scripts/reproduce_kit.sh`, confirmed by `replication.py`): a stranger runs one
  hermetic image to regenerate every byte-reproducible table and re-confirm every
  oracle against the real toolchain, ending with a content-hash manifest whose
  `kit_hash` (over corpus stats, the tool-applicability table and kit file hashes)
  reproduces identically across runs — the diffability an artifact evaluator needs
- a **statistical-rigor study** (`statistical_rigor.py`): the real definedness
  oracle is run over seeded subsamples across multiple seeds, reporting recall
  and false-positive-rate with **pre-registered metrics** and **Wilson 95 %
  confidence intervals** computed deterministically from the counts, zero false
  positives on the equivalent population (the sound-for-divergence guarantee,
  measured), honest `out_of_scope` accounting of value divergences, and a
  content-hashed outcome layer that reproduces identically across runs
- a **relational product-program formalization** (`product_program.py`, spec in
  `docs/PRODUCT_PROGRAM.md`): the oracle recast as **cross-language translation
  validation** — a product program `P_S × P_T` carrying a relational assertion
  `R_m` whose violation is exactly a divergence, given as inference rules with a
  soundness-and-relative-completeness theorem parameterized over the target
  semantics pack, discharged both by an exhaustive Boolean-equivalence check
  against the operational semantics and on real clang/UBSan + rustc/go runs
- a **cross-language translation-validation interface** (`translation_validation.py`,
  spec in `docs/TRANSLATION_VALIDATION.md`): the oracle exposed as a per-instance,
  witness-producing validator that `REFUTES` a producer's faithfulness claim with
  a **re-executable counterexample witness** (carrying both source texts + the
  input) whose `replay()` recompiles both sides from scratch — proven on real
  code to replay to the same divergence (witness soundness) and deterministically
- a **generalization study** (`generalization.py`): the real oracle run over a
  grid of (language pair `rust`/`go`/`swift`) × (producer style `direct`/`helper`/
  `verbose`) × (divergence class) × (input), proving the divergence result is
  **invariant across every cell** — detection rate 1.0 on UB inputs and zero
  false positives on safe inputs for every pair and style — so the result is not
  an artefact of one pair, one transpiler, or one input distribution
- an **ACM artifact-evaluation harness** (`artifact_eval.py`): the criteria for
  the **Available / Functional / Reproduced** badges are encoded as *checked
  predicates* — open licence + citation descriptor + version consistency
  (Available), self-documentation + exercisable entry points + a **live** real-
  oracle smoke-test (Functional), and byte-identical results plus stable
  replication/scale/generalization reproducibility hashes (Reproduced) — so
  "we earn the badges" is itself an entry in the traceability matrix (see
  `docs/ARTIFACT.md`)
- a **machine-checked soundness proof** (`formal/ProductSoundness.lean`,
  driver `mechanized_soundness.py`): the product-program decision procedure's
  core guarantees — soundness (no false alarms), relative completeness,
  "every counterexample is rooted in source UB", and the safety corollary that
  equivalent pairs are never flagged — are proven in **Lean 4** for a
  language-pair-parametric calculus instantiated to C→Rust, and the real Lean
  kernel is invoked to confirm it (see `docs/MECHANIZED_SOUNDNESS.md`)
- a **Tier-2 idiomatic anchor corpus** (`idiomatic_corpus.py`): realistic,
  value-carrying ported functions — the binary-search **midpoint `(lo+hi)/2`**
  overflow bug, a packed-struct bit-field shift, a coreutils-style rate divide,
  and their equivalent idiomatic fixes (64-bit-widened average, byte clamp,
  additive checksum) — on which the oracle is exactly right across rust and go:
  every divergent port flagged, every equivalent port left silent
- a **Tier-3 multi-pair corpus** (`multipair_corpus.py`): each real C function
  translated to **all three** targets (Rust/Go/Swift) in transpiler/LLM style,
  proving **cross-pair invariance** — a UB-rooted divergence is flagged on every
  pair (it's a property of the source UB, not one target's quirks) and an
  equivalent function on none (15 function×pair verdicts, all correct)
- **pluggable transpiler-integration recipes** (`transpiler_recipes.py`,
  `docs/TRANSPILER_RECIPES.md`) that realise the tool's workflow —
  *translate your C with `$tool`, then verify with us*: a `Translator` protocol
  is the integration point, built-in `ReferenceTranslator`s ship compilable
  Rust/Go/Swift baselines, and `ExternalCommandTranslator` shells out to a real
  transpiler (c2rust, or an LLM-transpiler CLI) **gated** on the binary existing
  — absent, it reports unavailable and fabricates nothing. On every available
  pair the translate→verify step flags a div-by-zero divergence on the UB input
  and stays silent on a safe one, proving the recipe preserves the oracle's
  guarantees end to end
- a **frozen shared-IR contract** (`ir.py`, spec in `docs/IR.md`): the single
  language-pair-agnostic translation-unit shape every frontend lowers into and
  every oracle consumes, plus a validator that **rejects ill-formed lowerings**
  at the manifest boundary (the CLI exits non-zero on a malformed unit) while
  still accepting well-formed-but-uncovered units as `NOT_COVERED`
- a **machine-checked theory↔implementation traceability map**
  (`traceability.py`, generated table in `docs/TRACEABILITY.md`): every claim the
  project makes is bound to a named module + symbols, and — where the claim is a
  checkable theorem — to a fast runnable core, so `verify_traceability()` returns
  `[]` only when every module imports, every cited symbol exists, and every
  attached theorem passes (artifact evaluators can re-run it in one line)
- a **formal divergence semantics** (`semantics.py`, spec in `docs/SEMANTICS.md`):
  the property the tool witnesses — *divergence modulo source-undefinedness* — is
  given as an executable predicate (premise: source UB is reached; consequence:
  optimizer-exploited value flip, or a trap-vs-defined gap; target defined), and
  a checked theorem proves it **coincides exactly** with the re-execution
  harness's `confirmed` decision on real compiled programs
- a **per-class completeness characterization** (`completeness.py`, spec in
  `docs/COMPLETENESS.md`): for the integer classes the symbolic search is proven
  not just sound but **complete on a precisely-stated bounded fragment** — an
  executable check enumerates ground truth by brute force and asserts the oracle
  reports a witness *exactly* when one exists (no false negatives), across every
  registered language pair, with one diverging unit per class confirmed end-to-end
  against real compilers
- a **positioning vs adjacent verifiers** ([`docs/POSITIONING.md`](docs/POSITIONING.md)):
  a precise account of the cross-language UB-rooted-divergence gap that BMC for C
  (CBMC/ESBMC), same-language equivalence/translation validators, target-language
  verifiers (Kani/Prusti/Miri) and undirected differential fuzzing each leave open
- benchmark assets and sample projects under `examples/`

## Best way to use this checkout

Treat it as a **research-grade verifier and exploration harness** for cross-
language equivalence. The implementation is substantial and the CLI surface is
real; the responsible story is about executable analysis workflows, not about
unrerun leaderboard-style claims.
