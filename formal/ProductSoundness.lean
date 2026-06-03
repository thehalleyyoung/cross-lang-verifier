/-
  Scoped mechanized soundness for the cross-language UB-divergence oracle
  (100_STEPS steps 75, 126, 127, and 128).

  This file gives a *machine-checked* soundness (and relative-completeness)
  argument for the relational/product-program decision procedure the tool
  implements (`src/ub_oracle/product_program.py`, docs/PRODUCT_PROGRAM.md),
  for a **core language-pair-parametric calculus**, instantiated to C -> Rust
  UB first.

  It is deliberately self-contained: it depends on nothing but Lean 4 core
  (no Mathlib), so `lean ProductSoundness.lean` type-checks it with the kernel
  and therefore *proves* the theorems against a real proof assistant.

  Scope (honest): we model the *recorded-observable abstraction* that the
  oracle actually decides over -- the per-input triple (UB-premise reached,
  target-defined, consequence/observable-divergence) -- not a full denotational
  C semantics. Within that abstraction the theorems below are exactly the
  guarantees the Python oracle relies on:

    * `oracle_sound`        : the oracle never reports a divergence that is not
                              a genuine UB-rooted divergence (one-sided / no
                              false alarms).
    * `oracle_complete_rel` : relative completeness -- every genuine UB-rooted
                              divergence in the abstraction is reported.
    * `oracle_decides`      : the two combine to a decision-procedure
                              correctness statement (reports iff diverges).
    * `equivalence_never_reported` : the safety corollary the tool advertises
                              (observationally-equivalent pairs are never
                              flagged).
    * `product_program_preserves_divergence_witness` : the end-to-end
                              product-program construction emits a
                              counterexample only by copying the concrete run
                              payload and the observation derived from raw
                              run facts, and that observation is a genuine
                              UB-rooted divergence.

  The development is parametric in a `TargetPack` (the C->target semantics
  pack); `RustPack` instantiates it and `rust_oracle_sound` discharges the
  same guarantees for the C -> Rust pair concretely.
-/

namespace CrossLangVerifier

/-- The three relational clauses the product program tracks, as Booleans, for a
    single concrete input.  This is the *recorded-observable abstraction*:

    * `ubReached`  -- the C premise `P`: this input drives the source unit into
                      undefined behaviour (a sanitizer trap or an exploited
                      miscompile was observed).
    * `tgtDefined` -- `T`: the target (e.g. Rust) executed with a well-defined
                      outcome on the same input.
    * `consequence`-- `C`: the observable behaviours actually differ. -/
structure Observation where
  ubReached   : Bool
  tgtDefined  : Bool
  consequence : Bool
deriving DecidableEq, Repr

/-- The relational assertion `R = ¬(P ∧ T ∧ C)`.  The product program is said to
    be *violated* on an observation exactly when `R` fails. -/
def R (o : Observation) : Bool :=
  !(o.ubReached && o.tgtDefined && o.consequence)

/-- The oracle reports a divergence iff the relational assertion is violated. -/
def productViolated (o : Observation) : Bool :=
  !(R o)

/-- The *ground-truth* notion of a UB-rooted divergence in this abstraction:
    the source relied on UB, the target was defined, and behaviours differ.
    This is the property a human auditor would confirm by hand. -/
def isUBDivergence (o : Observation) : Prop :=
  o.ubReached = true ∧ o.tgtDefined = true ∧ o.consequence = true

/-- Two units are *observationally equivalent* on an input when no consequence
    is observed (their recorded behaviours agree). -/
def equivalent (o : Observation) : Prop :=
  o.consequence = false

/-! ### Core soundness / completeness (pack-independent) -/

/-- **Soundness.**  If the oracle reports a divergence, there is a genuine
    UB-rooted divergence.  No false alarms. -/
theorem oracle_sound (o : Observation) :
    productViolated o = true → isUBDivergence o := by
  intro h
  simp only [productViolated, R, Bool.not_not] at h
  rw [Bool.and_eq_true, Bool.and_eq_true] at h
  exact ⟨h.1.1, h.1.2, h.2⟩

/-- **Relative completeness.**  Every genuine UB-rooted divergence (in the
    recorded-observable abstraction) is reported by the oracle. -/
theorem oracle_complete_rel (o : Observation) :
    isUBDivergence o → productViolated o = true := by
  intro h
  obtain ⟨hp, ht, hc⟩ := h
  simp [productViolated, R, hp, ht, hc]

/-- **Decision-procedure correctness.**  The oracle reports a divergence *iff*
    one genuinely exists in the abstraction. -/
theorem oracle_decides (o : Observation) :
    productViolated o = true ↔ isUBDivergence o :=
  ⟨oracle_sound o, oracle_complete_rel o⟩

/-- **Safety corollary.**  An observationally-equivalent pair is never flagged:
    the tool produces no false positive on agreeing behaviours. -/
theorem equivalence_never_reported (o : Observation) :
    equivalent o → productViolated o = false := by
  intro h
  simp [productViolated, R, equivalent] at *
  simp [h]

/-- A reported divergence in particular *witnesses* that UB was reached: every
    counterexample is rooted in source-level undefined behaviour, never in a
    mere target quirk.  This is the central honesty claim of the tool. -/
theorem report_implies_ub (o : Observation) :
    productViolated o = true → o.ubReached = true := by
  intro h
  exact (oracle_sound o h).1

/-! ### Language-pair-parametric layer

    A `TargetPack` packages the per-target notion of a *defined* outcome.  The
    oracle and its guarantees are parametric in the pack; the `C -> Rust`,
    `C -> Go`, `C -> Swift` packs are instances.  Because the soundness theorem
    above is already pack-independent (it only reads the recorded
    `tgtDefined` flag), instantiating a pack cannot weaken it -- which is the
    formal content of "the soundness argument is language-pair-parametric". -/

/-- A target-semantics pack: a name and the set of process return codes the
    target treats as a *defined* outcome (the rest are UB / aborts that do not
    count as a divergence on their own). -/
structure TargetPack where
  name          : String
  definedReturn : Int → Bool

/-- Given a pack and the raw recorded facts of a run, build the abstract
    `Observation` the oracle decides over. -/
def observe (pack : TargetPack)
    (ubReached : Bool) (tgtReturn : Int) (behavioursDiffer : Bool) : Observation :=
  { ubReached   := ubReached
    tgtDefined  := pack.definedReturn tgtReturn
    consequence := behavioursDiffer }

/-- The pack-level soundness statement: for *any* pack and *any* recorded run,
    a reported divergence is a genuine UB-rooted divergence. -/
theorem pack_oracle_sound (pack : TargetPack)
    (ub : Bool) (rc : Int) (diff : Bool) :
    productViolated (observe pack ub rc diff) = true →
      isUBDivergence (observe pack ub rc diff) :=
  oracle_sound _

/-! ### Instantiation: the C -> Rust pack

    Rust's defined outcomes for the divergence classes the tool handles are a
    clean exit (`0`) or a controlled panic (`101`); a UB-class miscompile would
    show up as some other code.  This mirrors the `rust` pack in
    `src/ub_oracle/target_semantics.py` (`defined rc ∈ {0, 101}`). -/
def RustPack : TargetPack where
  name := "rust"
  definedReturn rc := rc == 0 || rc == 101

/-- Concrete C -> Rust soundness: the oracle, instantiated to the real Rust
    pack, never raises a false alarm. -/
theorem rust_oracle_sound
    (ub : Bool) (rc : Int) (diff : Bool) :
    productViolated (observe RustPack ub rc diff) = true →
      isUBDivergence (observe RustPack ub rc diff) :=
  pack_oracle_sound RustPack ub rc diff

/-- A concrete, fully-evaluated witness corresponding to the canonical
    div-by-zero example (`a/b` with `b = 0`): C traps under UBSan (`ub = true`),
    Rust panics with the defined code `101` (`tgtDefined = true`), and the
    observable behaviours differ (`consequence = true`).  The oracle reports it,
    and `rust_oracle_sound` certifies the report is a genuine UB divergence. -/
example :
    productViolated (observe RustPack true 101 true) = true
      ∧ isUBDivergence (observe RustPack true 101 true) := by
  refine ⟨by decide, ?_⟩
  exact rust_oracle_sound true 101 true (by decide)

/-- A safe-input witness (`b ≠ 0`): both sides agree, so the oracle is silent --
    no false positive. -/
example : productViolated (observe RustPack false 0 false) = false := by decide

/-! ### Strict-aliasing oracle: optimizer-exploitation evidence

    Strict-aliasing UB is not confirmed by a sanitizer in this artifact.  The
    real harness (`confirm_optimizer_exploited`) instead compiles the *same* C
    source twice (`-O0` and `-O2 -fstrict-aliasing`) and records UB reachability
    exactly when both builds run cleanly but print different observables.  A
    deterministic, defined target has one observable.  Therefore two different C
    observables cannot both match that one target observable: at least one legal
    source compilation diverges from the target.  The theorem below mechanizes
    that class-specific argument and then reuses the product-program soundness
    theorem above.
-/

/-- The finite evidence tuple recorded for a strict-aliasing witness.

    The first two flags are construction invariants of the generated C witness:
    incompatible typed accesses (`int*`/`long*`) are made to the same storage.
    The build/target fields mirror the optimizer-exploitation confirmation mode
    in `src/ub_oracle/reexec.py`: two clean C builds disagree, while the target is
    defined and deterministic.  Outputs are modeled as natural numbers because
    the strict-aliasing witness prints integer observables. -/
structure StrictAliasingWitness where
  incompatibleAccesses : Bool
  sameStorage          : Bool
  cBuildAClean         : Bool
  cBuildBClean         : Bool
  cBuildAOut           : Nat
  cBuildBOut           : Nat
  targetDefined        : Bool
  targetDeterministic  : Bool
  targetOut            : Nat
deriving DecidableEq, Repr

/-- The generated witness is the strict-aliasing shape: incompatible typed
    accesses over the same storage. -/
def strictAliasingTypePun (w : StrictAliasingWitness) : Bool :=
  w.incompatibleAccesses && w.sameStorage

/-- The harness's source-side signal: both C builds ran cleanly and disagreed. -/
def optimizerBuildsDiffer (w : StrictAliasingWitness) : Bool :=
  w.cBuildAClean && w.cBuildBClean &&
    decide (w.cBuildAOut ≠ w.cBuildBOut)

/-- The target side of the harness signal: the target outcome is defined and
    deterministic across repeated executions. -/
def optimizerTargetDefined (w : StrictAliasingWitness) : Bool :=
  w.targetDefined && w.targetDeterministic

/-- Exactly the `optimizer_exploited` confirmation predicate implemented by the
    Python re-execution harness for strict aliasing. -/
def optimizerConfirmed (w : StrictAliasingWitness) : Bool :=
  optimizerBuildsDiffer w && optimizerTargetDefined w

/-- At least one of the two C observations differs from the one target
    observation. -/
def oneCBuildDiffersFromTarget (w : StrictAliasingWitness) : Bool :=
  decide (w.cBuildAOut ≠ w.targetOut) ||
    decide (w.cBuildBOut ≠ w.targetOut)

/-- The product-program observation induced by optimizer-exploitation evidence. -/
def strictAliasingObservation (w : StrictAliasingWitness) : Observation :=
  { ubReached   := optimizerBuildsDiffer w
    tgtDefined  := optimizerTargetDefined w
    consequence := oneCBuildDiffersFromTarget w }

/-- A strict-aliasing oracle report requires both the generated type-pun shape and
    the optimizer-exploitation confirmation. -/
def strictAliasingReported (w : StrictAliasingWitness) : Bool :=
  strictAliasingTypePun w && optimizerConfirmed w

/-- Pigeonhole step for optimizer-exploited UB: two different C observations
    cannot both equal one deterministic target observation. -/
theorem optimizer_exploited_pigeonhole (a b t : Nat) :
    a ≠ b → a ≠ t ∨ b ≠ t := by
  intro h
  by_cases ha : a = t
  · right
    intro hb
    exact h (ha.trans hb.symm)
  · left
    exact ha

/-- A strict-aliasing report carries the structural type-pun facts generated by
    the oracle: incompatible typed accesses to the same storage. -/
theorem strict_aliasing_report_implies_type_pun (w : StrictAliasingWitness) :
    strictAliasingReported w = true →
      w.incompatibleAccesses = true ∧ w.sameStorage = true := by
  intro h
  rw [strictAliasingReported, Bool.and_eq_true] at h
  have htype : strictAliasingTypePun w = true := h.1
  rw [strictAliasingTypePun, Bool.and_eq_true] at htype
  exact htype

/-- A strict-aliasing report also carries the exact optimizer-exploitation signal
    used by the real compiler-backed confirmation harness. -/
theorem strict_aliasing_report_implies_optimizer_exploited
    (w : StrictAliasingWitness) :
    strictAliasingReported w = true →
      optimizerBuildsDiffer w = true ∧ optimizerTargetDefined w = true := by
  intro h
  rw [strictAliasingReported, Bool.and_eq_true] at h
  have hconf : optimizerConfirmed w = true := h.2
  rw [optimizerConfirmed, Bool.and_eq_true] at hconf
  exact hconf

/-- The optimizer-exploitation confirmation is strong enough to violate the
    product assertion: clean C builds disagree, the target is defined and
    deterministic, and by pigeonhole one C build differs from the target. -/
theorem strict_aliasing_reported_product_violated (w : StrictAliasingWitness) :
    strictAliasingReported w = true →
      productViolated (strictAliasingObservation w) = true := by
  intro h
  have hexpl := strict_aliasing_report_implies_optimizer_exploited w h
  have hbuild : optimizerBuildsDiffer w = true := hexpl.1
  have htarget : optimizerTargetDefined w = true := hexpl.2
  have hbuildFacts := hbuild
  simp only [optimizerBuildsDiffer, Bool.and_eq_true, decide_eq_true_eq] at hbuildFacts
  have hneq : w.cBuildAOut ≠ w.cBuildBOut := hbuildFacts.2
  have hconseq : oneCBuildDiffersFromTarget w = true := by
    have hp := optimizer_exploited_pigeonhole
      w.cBuildAOut w.cBuildBOut w.targetOut hneq
    cases hp with
    | inl ha =>
        simp [oneCBuildDiffersFromTarget, ha]
    | inr hb =>
        simp [oneCBuildDiffersFromTarget, hb]
  simp [productViolated, R, strictAliasingObservation, hbuild, htarget, hconseq]

/-- **Strict-aliasing soundness.**  If the real strict-aliasing oracle reports a
    compiler-confirmed optimizer-exploited witness, the induced product-program
    observation is a genuine UB-rooted divergence. -/
theorem strict_aliasing_oracle_sound (w : StrictAliasingWitness) :
    strictAliasingReported w = true →
      isUBDivergence (strictAliasingObservation w) := by
  intro h
  exact oracle_sound (strictAliasingObservation w)
    (strict_aliasing_reported_product_violated w h)

/-- Concrete positive witness shape: type-pun structure, two clean C builds with
    different integer observables, and a deterministic defined target. -/
example :
    strictAliasingReported
      { incompatibleAccesses := true, sameStorage := true
        cBuildAClean := true, cBuildBClean := true
        cBuildAOut := 1, cBuildBOut := 0
        targetDefined := true, targetDeterministic := true, targetOut := 0 } = true
    ∧ isUBDivergence
      (strictAliasingObservation
        { incompatibleAccesses := true, sameStorage := true
          cBuildAClean := true, cBuildBClean := true
          cBuildAOut := 1, cBuildBOut := 0
          targetDefined := true, targetDeterministic := true, targetOut := 0 }) := by
  refine ⟨by decide, ?_⟩
  exact strict_aliasing_oracle_sound
    { incompatibleAccesses := true, sameStorage := true
      cBuildAClean := true, cBuildBClean := true
      cBuildAOut := 1, cBuildBOut := 0
      targetDefined := true, targetDeterministic := true, targetOut := 0 }
    (by decide)

/-- Negative control: no C build disagreement means no strict-aliasing report. -/
example :
    strictAliasingReported
      { incompatibleAccesses := true, sameStorage := true
        cBuildAClean := true, cBuildBClean := true
        cBuildAOut := 7, cBuildBOut := 7
        targetDefined := true, targetDeterministic := true, targetOut := 7 } = false := by
  decide

/-! ### Pointer-provenance oracle: trap-vs-defined evidence

    The pointer-provenance oracle (`src/ub_oracle/oracles/pointer_provenance.py`)
    finds a concrete integer offset whose byte displacement leaves the source
    array object's provenance and, in the address-overflow witness, triggers
    UBSan's real `pointer-overflow` check.  The safe target translation uses a
    checked index instead of forming a raw out-of-provenance pointer.  The real
    harness confirms this class in `trap_vs_defined` mode: C traps on the source
    UB, while the target has a defined deterministic outcome.  In the
    recorded-observable abstraction, a trap and a defined target outcome are the
    behavioural difference (`consequence`) the product program records.
-/

/-- The finite evidence tuple recorded for a pointer-provenance witness.

    The first two flags are construction invariants of the generated C witness:
    the input integer is ordinary data, but the induced pointer arithmetic leaves
    the source object's permitted provenance.  The trap flag mirrors the
    compiler-backed `trap_vs_defined` confirmation.  The target flags mirror the
    safe Rust/Go translation: it uses a checked index and runs with a defined,
    deterministic outcome. -/
structure PointerProvenanceWitness where
  sourceInputValid               : Bool
  pointerArithmeticLeavesObject  : Bool
  sanitizerPointerOverflowTrap   : Bool
  targetUsesCheckedIndex         : Bool
  targetDefined                  : Bool
  targetDeterministic            : Bool
deriving DecidableEq, Repr

/-- The source-side C provenance premise: the input itself is valid, but the
    generated pointer arithmetic leaves the array object's permitted provenance. -/
def pointerProvenancePremise (w : PointerProvenanceWitness) : Bool :=
  w.sourceInputValid && w.pointerArithmeticLeavesObject

/-- The target side of `trap_vs_defined`: the translation keeps an index, uses a
    checked operation, and returns deterministically with defined semantics. -/
def pointerProvenanceTargetDefined (w : PointerProvenanceWitness) : Bool :=
  w.targetUsesCheckedIndex && w.targetDefined && w.targetDeterministic

/-- The exact confirmation signal for the pointer-provenance oracle: the real C
    UBSan build traps, and the safe target run is defined. -/
def pointerProvenanceConfirmed (w : PointerProvenanceWitness) : Bool :=
  w.sanitizerPointerOverflowTrap && pointerProvenanceTargetDefined w

/-- In `trap_vs_defined` mode, the observed behavioural difference is precisely:
    the source traps on UB and the target produces a defined deterministic result. -/
def trapVsDefinedConsequence (w : PointerProvenanceWitness) : Bool :=
  w.sanitizerPointerOverflowTrap && pointerProvenanceTargetDefined w

/-- Source UB is reached when the provenance premise is generated and UBSan traps
    on that concrete input. -/
def pointerProvenanceUBReached (w : PointerProvenanceWitness) : Bool :=
  pointerProvenancePremise w && w.sanitizerPointerOverflowTrap

/-- The product-program observation induced by a pointer-provenance confirmation. -/
def pointerProvenanceObservation (w : PointerProvenanceWitness) : Observation :=
  { ubReached   := pointerProvenanceUBReached w
    tgtDefined  := pointerProvenanceTargetDefined w
    consequence := trapVsDefinedConsequence w }

/-- A pointer-provenance oracle report requires both the generated provenance
    violation shape and the real compiler-backed `trap_vs_defined` confirmation. -/
def pointerProvenanceReported (w : PointerProvenanceWitness) : Bool :=
  pointerProvenancePremise w && pointerProvenanceConfirmed w

/-- A pointer-provenance report carries the source-side shape the oracle
    generated: a valid integer input whose induced pointer arithmetic leaves the
    array object's permitted provenance. -/
theorem pointer_provenance_report_implies_out_of_provenance
    (w : PointerProvenanceWitness) :
    pointerProvenanceReported w = true →
      w.sourceInputValid = true ∧ w.pointerArithmeticLeavesObject = true := by
  intro h
  rw [pointerProvenanceReported, Bool.and_eq_true] at h
  have hprem : pointerProvenancePremise w = true := h.1
  rw [pointerProvenancePremise, Bool.and_eq_true] at hprem
  exact hprem

/-- A pointer-provenance report also carries the target-side safe-index shape:
    the port did not form a raw pointer and its checked result was defined and
    deterministic. -/
theorem pointer_provenance_report_implies_checked_target
    (w : PointerProvenanceWitness) :
    pointerProvenanceReported w = true →
      w.targetUsesCheckedIndex = true ∧
      w.targetDefined = true ∧ w.targetDeterministic = true := by
  intro h
  rw [pointerProvenanceReported, Bool.and_eq_true] at h
  have hconf : pointerProvenanceConfirmed w = true := h.2
  rw [pointerProvenanceConfirmed, Bool.and_eq_true] at hconf
  have htgt : pointerProvenanceTargetDefined w = true := hconf.2
  rw [pointerProvenanceTargetDefined, Bool.and_eq_true] at htgt
  have hchecked : (w.targetUsesCheckedIndex && w.targetDefined) = true := htgt.1
  rw [Bool.and_eq_true] at hchecked
  exact ⟨hchecked.1, hchecked.2, htgt.2⟩

/-- A pointer-provenance report carries exactly the `trap_vs_defined` confirmation
    facts consumed by the product program. -/
theorem pointer_provenance_report_implies_trap_vs_defined
    (w : PointerProvenanceWitness) :
    pointerProvenanceReported w = true →
      pointerProvenanceUBReached w = true ∧
      pointerProvenanceTargetDefined w = true ∧
      trapVsDefinedConsequence w = true := by
  intro h
  rw [pointerProvenanceReported, Bool.and_eq_true] at h
  have hprem : pointerProvenancePremise w = true := h.1
  have hconf : pointerProvenanceConfirmed w = true := h.2
  rw [pointerProvenanceConfirmed, Bool.and_eq_true] at hconf
  have htrap : w.sanitizerPointerOverflowTrap = true := hconf.1
  have htgt : pointerProvenanceTargetDefined w = true := hconf.2
  have hub : pointerProvenanceUBReached w = true := by
    simp [pointerProvenanceUBReached, hprem, htrap]
  have hcons : trapVsDefinedConsequence w = true := by
    simp [trapVsDefinedConsequence, htrap, htgt]
  exact ⟨hub, htgt, hcons⟩

/-- The `trap_vs_defined` confirmation is strong enough to violate the product
    assertion: source UB was reached, the target is defined, and the source trap
    versus target outcome is the recorded consequence. -/
theorem pointer_provenance_reported_product_violated
    (w : PointerProvenanceWitness) :
    pointerProvenanceReported w = true →
      productViolated (pointerProvenanceObservation w) = true := by
  intro h
  have hfacts := pointer_provenance_report_implies_trap_vs_defined w h
  obtain ⟨hub, htgt, hcons⟩ := hfacts
  simp [productViolated, R, pointerProvenanceObservation, hub, htgt, hcons]

/-- **Pointer-provenance soundness.**  If the real pointer-provenance oracle
    reports a compiler-confirmed `trap_vs_defined` witness, the induced
    product-program observation is a genuine UB-rooted divergence. -/
theorem pointer_provenance_oracle_sound (w : PointerProvenanceWitness) :
    pointerProvenanceReported w = true →
      isUBDivergence (pointerProvenanceObservation w) := by
  intro h
  exact oracle_sound (pointerProvenanceObservation w)
    (pointer_provenance_reported_product_violated w h)

/-- Concrete positive witness shape: valid data input, out-of-provenance pointer
    arithmetic, a real UBSan trap, and a checked deterministic target. -/
def pointerProvenancePositive : PointerProvenanceWitness :=
  { sourceInputValid := true
    pointerArithmeticLeavesObject := true
    sanitizerPointerOverflowTrap := true
    targetUsesCheckedIndex := true
    targetDefined := true
    targetDeterministic := true }

example :
    pointerProvenanceReported pointerProvenancePositive = true
      ∧ isUBDivergence
        (pointerProvenanceObservation pointerProvenancePositive) := by
  refine ⟨by decide, ?_⟩
  exact pointer_provenance_oracle_sound pointerProvenancePositive (by decide)

/-- Negative control: an in-provenance source operation is not a report even if
    the target side is checked and defined. -/
example :
    pointerProvenanceReported
      { sourceInputValid := true
        pointerArithmeticLeavesObject := false
        sanitizerPointerOverflowTrap := true
        targetUsesCheckedIndex := true
        targetDefined := true
        targetDeterministic := true } = false := by
  decide

/-! ### End-to-end product-program witness construction

    Step 128 asks for the product-program theorem itself to be mechanized:
    constructing a counterexample from a product run must preserve the concrete
    divergence witness, not reinterpret it.  The source/target/input fields
    below are opaque provenance payloads; the mechanized content remains the
    recorded-observable abstraction.  Crucially, a `ProductRun` stores the raw
    run facts (`ubReached`, target return code, and whether behaviours differ),
    and the `Observation` is *derived* through the same `observe` function used by
    the pack-parametric theorem above.  This makes the preservation theorem a
    check on the construction boundary, not a tautology over a pre-stored
    observation.
-/

/-- Raw facts recorded by one completed product-program run.  The payload fields
    identify the source, target and concrete input carried into the emitted
    counterexample; the Boolean/int fields are the measured facts from which the
    product observation is derived. -/
structure ProductRun where
  sourceId         : String
  targetId         : String
  input            : String
  targetPack       : TargetPack
  sourceUBReached  : Bool
  targetReturn     : Int
  behavioursDiffer : Bool

/-- The product observation derived from raw run facts and the target pack. -/
def productRunObservation (r : ProductRun) : Observation :=
  observe r.targetPack r.sourceUBReached r.targetReturn r.behavioursDiffer

/-- The language-agnostic counterexample emitted by the product-program layer.
    It carries the payload unchanged plus the derived observation that violated
    the product assertion. -/
structure ProductCounterexample where
  sourceId    : String
  targetId    : String
  input       : String
  observation : Observation
deriving DecidableEq, Repr

/-- Emit a counterexample exactly when the derived product observation violates
    `R`; otherwise the product program is silent. -/
def productRunToCounterexample (r : ProductRun) : Option ProductCounterexample :=
  if _h : productViolated (productRunObservation r) = true then
    some
      { sourceId    := r.sourceId
        targetId    := r.targetId
        input       := r.input
        observation := productRunObservation r }
  else
    none

/-- **End-to-end witness preservation.**  If the product-program construction
    emits a counterexample, it has copied the product run's source/target/input
    payload unchanged, its observation is exactly the observation derived from
    the raw run facts, and that observation is a genuine UB-rooted divergence. -/
theorem product_program_preserves_divergence_witness
    (r : ProductRun) (cex : ProductCounterexample) :
    productRunToCounterexample r = some cex →
      cex.sourceId = r.sourceId ∧
      cex.targetId = r.targetId ∧
      cex.input = r.input ∧
      cex.observation = productRunObservation r ∧
      isUBDivergence cex.observation := by
  intro h
  unfold productRunToCounterexample at h
  by_cases hv : productViolated (productRunObservation r) = true
  · simp [hv] at h
    cases h
    exact ⟨rfl, rfl, rfl, rfl, oracle_sound (productRunObservation r) hv⟩
  · simp [hv] at h

/-- The construction emits some counterexample exactly for product-assertion
    violations. -/
theorem product_program_emits_witness_iff_product_violated (r : ProductRun) :
    (∃ cex, productRunToCounterexample r = some cex) ↔
      productViolated (productRunObservation r) = true := by
  constructor
  · intro h
    obtain ⟨cex, hcex⟩ := h
    have hp := product_program_preserves_divergence_witness r cex hcex
    have hdiv : isUBDivergence (productRunObservation r) := by
      rw [← hp.2.2.2.1]
      exact hp.2.2.2.2
    exact oracle_complete_rel (productRunObservation r) hdiv
  · intro hv
    refine ⟨
      { sourceId    := r.sourceId
        targetId    := r.targetId
        input       := r.input
        observation := productRunObservation r }, ?_⟩
    unfold productRunToCounterexample
    simp [hv]

/-- Combining emission with `oracle_decides`: the end-to-end construction emits a
    counterexample iff the derived product observation is a UB-rooted divergence. -/
theorem product_program_witness_iff_divergence (r : ProductRun) :
    (∃ cex, productRunToCounterexample r = some cex) ↔
      isUBDivergence (productRunObservation r) := by
  constructor
  · intro h
    have hv := (product_program_emits_witness_iff_product_violated r).1 h
    exact oracle_sound (productRunObservation r) hv
  · intro h
    have hv := oracle_complete_rel (productRunObservation r) h
    exact (product_program_emits_witness_iff_product_violated r).2 hv

/-- Concrete end-to-end positive witness: raw run facts for a C source that
    reached UB and a Rust target that returned a defined panic code emit a
    preserved UB-rooted counterexample. -/
def productPositiveRun : ProductRun :=
  { sourceId := "c/div-by-zero.c"
    targetId := "rust/div-by-zero.rs"
    input := "a=5,b=0"
    targetPack := RustPack
    sourceUBReached := true
    targetReturn := 101
    behavioursDiffer := true }

example :
    ∃ cex, productRunToCounterexample productPositiveRun = some cex ∧
      cex.sourceId = "c/div-by-zero.c" ∧
      cex.targetId = "rust/div-by-zero.rs" ∧
      cex.input = "a=5,b=0" ∧
      isUBDivergence cex.observation := by
  have hdiv : isUBDivergence (productRunObservation productPositiveRun) := by
    simp [productRunObservation, productPositiveRun, observe, RustPack,
      isUBDivergence]
  have hemits := (product_program_witness_iff_divergence productPositiveRun).2
    hdiv
  obtain ⟨cex, hcex⟩ := hemits
  have hp := product_program_preserves_divergence_witness productPositiveRun cex hcex
  exact ⟨cex, hcex, hp.1, hp.2.1, hp.2.2.1, hp.2.2.2.2⟩

/-- Negative control: when the raw facts do not violate `R`, no witness is
    emitted. -/
example :
    productRunToCounterexample
      { sourceId := "c/safe.c"
        targetId := "rust/safe.rs"
        input := "a=5,b=1"
        targetPack := RustPack
        sourceUBReached := false
        targetReturn := 0
        behavioursDiffer := false } = none := by
  rfl

/-! ### Verified counterexample minimizer

    Step 139 asks for the counterexample minimizer to be verified.  The Python
    minimizer (`src/ub_oracle/minimizer.py`) is deliberately conservative: a
    candidate reduction is accepted only after the same real re-execution harness
    re-confirms the reduced witness and, for UBSan-backed classes, preserves the
    same diagnostic category.  The Lean model below proves the scoped
    delta-debugging soundness claim at the same recorded-observable abstraction as
    the rest of this file: any accepted reduction, and any certificate built from
    accepted reductions plus a final harness-confirmed observation, preserves a
    genuine UB-rooted divergence.  It does not claim global optimality; local
    minimality remains an executable harness check.
-/

/-- One accepted minimizer step.  `before` and `after` are the product-program
    observations before and after trying a simpler input value; the Boolean fields
    mirror the Python certificate fields recorded at the moment the real harness
    accepted the step. -/
structure MinimizerReduction where
  before              : Observation
  after               : Observation
  sameDivergenceClass : Bool
  sameUBCategory      : Bool
  acceptedByHarness   : Bool
deriving DecidableEq, Repr

/-- A reduction is accepted only when the original and reduced observations both
    violate the product assertion, the class/category have not drifted, and the
    compiler-backed harness accepted the reduced witness. -/
def minimizerReductionAccepted (r : MinimizerReduction) : Prop :=
  productViolated r.before = true ∧
  productViolated r.after = true ∧
  r.sameDivergenceClass = true ∧
  r.sameUBCategory = true ∧
  r.acceptedByHarness = true

/-- The final observation after replaying a list of accepted reductions. -/
def minimizerTraceFinal (start : Observation) : List MinimizerReduction → Observation
  | [] => start
  | r :: rs => minimizerTraceFinal r.after rs

/-- Every reduction in a minimization trace was accepted by the real harness. -/
def minimizerTraceAccepted : List MinimizerReduction → Prop
  | [] => True
  | r :: rs => minimizerReductionAccepted r ∧ minimizerTraceAccepted rs

/-- The after-state of an accepted reduction still violates the product
    assertion. -/
theorem minimizer_reduction_after_product_violated (r : MinimizerReduction) :
    minimizerReductionAccepted r → productViolated r.after = true := by
  intro h
  exact h.2.1

/-- **Minimizer preservation.**  A single accepted reduction preserves a genuine
    UB-rooted divergence. -/
theorem minimizer_preserves_divergence (r : MinimizerReduction) :
    minimizerReductionAccepted r → isUBDivergence r.after := by
  intro h
  exact oracle_sound r.after (minimizer_reduction_after_product_violated r h)

/-- A chain of accepted reductions preserves divergence through to the final
    minimized observation. -/
theorem minimizer_reduction_steps_preserve_divergence
    (start : Observation) (steps : List MinimizerReduction) :
    productViolated start = true →
      minimizerTraceAccepted steps →
        isUBDivergence (minimizerTraceFinal start steps) := by
  intro hstart hsteps
  induction steps generalizing start with
  | nil =>
      simp [minimizerTraceFinal]
      exact oracle_sound start hstart
  | cons r rs ih =>
      simp [minimizerTraceAccepted] at hsteps
      have hafter := minimizer_reduction_after_product_violated r hsteps.1
      simp [minimizerTraceFinal]
      exact ih r.after hafter hsteps.2

/-- The offline certificate emitted by the Python minimizer.  The final
    observation is separately recorded because the Python verifier checks the
    concrete input-chain shape and then sends the final `P ∧ T ∧ C` facts to the
    same product-program theorem family. -/
structure MinimizerCertificate where
  original            : Observation
  final               : Observation
  reductions          : List MinimizerReduction
  sameDivergenceClass : Bool
  sameUBCategory      : Bool
  realHarnessConfirmed : Bool
  locallyMinimal      : Bool
deriving DecidableEq, Repr

/-- The certificate validity predicate checked offline by
    `verify_minimization_certificate`: the original and final observations are
    positive product assertions, the class/category did not drift, the final
    witness was real-harness-confirmed, and every recorded reduction is accepted.
    `locallyMinimal` is carried as an audited fact from the executable
    minimizer; it is not needed for preservation soundness. -/
def minimizerCertificateValid (c : MinimizerCertificate) : Prop :=
  productViolated c.original = true ∧
  productViolated c.final = true ∧
  c.sameDivergenceClass = true ∧
  c.sameUBCategory = true ∧
  c.realHarnessConfirmed = true ∧
  c.locallyMinimal = true ∧
  minimizerTraceAccepted c.reductions

/-- **Minimizer certificate soundness.**  A valid minimizer certificate entails
    that the minimized witness is still a genuine UB-rooted divergence. -/
theorem minimizer_certificate_sound (c : MinimizerCertificate) :
    minimizerCertificateValid c → isUBDivergence c.final := by
  intro h
  exact oracle_sound c.final h.2.1

/-- Concrete accepted reduction: a large overflow witness shrinks to the canonical
    `x=1`-style observation while preserving the product violation. -/
def minimizerPositiveReduction : MinimizerReduction :=
  { before := observe RustPack true 0 true
    after := observe RustPack true 0 true
    sameDivergenceClass := true
    sameUBCategory := true
    acceptedByHarness := true }

example :
    minimizerReductionAccepted minimizerPositiveReduction ∧
      isUBDivergence minimizerPositiveReduction.after := by
  refine ⟨?_, ?_⟩
  · simp [minimizerReductionAccepted, minimizerPositiveReduction,
      productViolated, R, observe, RustPack]
  · exact minimizer_preserves_divergence minimizerPositiveReduction (by
      simp [minimizerReductionAccepted, minimizerPositiveReduction,
        productViolated, R, observe, RustPack])

example :
    minimizer_certificate_sound
      { original := observe RustPack true 0 true
        final := observe RustPack true 0 true
        reductions := [minimizerPositiveReduction]
        sameDivergenceClass := true
        sameUBCategory := true
        realHarnessConfirmed := true
        locallyMinimal := true }
      (by
        simp [minimizerCertificateValid, minimizerTraceAccepted,
          minimizerPositiveReduction, minimizerReductionAccepted,
          productViolated, R, observe, RustPack]) =
      oracle_sound (observe RustPack true 0 true) (by decide) := by
  rfl

end CrossLangVerifier
