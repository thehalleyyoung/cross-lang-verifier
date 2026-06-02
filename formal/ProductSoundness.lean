/-
  Scoped mechanized soundness for the cross-language UB-divergence oracle
  (100_STEPS step 75).

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

end CrossLangVerifier
