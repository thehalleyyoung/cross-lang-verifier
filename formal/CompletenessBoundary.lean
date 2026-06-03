import ProductSoundness

/-
  Mechanized completeness boundary (100_STEPS step 131).

  ProductSoundness.lean proves the recorded-observable decision theorem:
  productViolated iff a UB-rooted divergence exists in that abstraction.  This
  file formalizes the *boundary* around the executable completeness claim: which
  divergence classes are guaranteed complete on their declared finite fragment,
  and which classes remain sound-but-may-abstain.

  Honest scope: this theorem does not replace the Python/Z3/brute-force
  completeness evidence in src/ub_oracle/completeness.py.  Instead it makes the
  advertised partition total, disjoint, and kernel-checked, and ties every
  in-fragment class back to ProductSoundness.oracle_decides.
-/

namespace CrossLangVerifier

/-- The divergence classes whose completeness status is currently part of the
    published boundary.  The string keys mirror src/ub_oracle/completeness.py. -/
inductive BoundaryClass where
  | signedOverflow
  | shiftOutOfRange
  | divisionByZero
  | intMinDivNeg1
  | arrayOob
  | strictAliasing
  | fpContraction
deriving DecidableEq, Repr

def classKey : BoundaryClass -> String
  | BoundaryClass.signedOverflow => "signed_overflow"
  | BoundaryClass.shiftOutOfRange => "shift_oob"
  | BoundaryClass.divisionByZero => "div_by_zero"
  | BoundaryClass.intMinDivNeg1 => "intmin_div_neg1"
  | BoundaryClass.arrayOob => "array_oob"
  | BoundaryClass.strictAliasing => "strict_aliasing"
  | BoundaryClass.fpContraction => "fp_contraction"

/-- Classes complete on the finite, declared operating-range fragment in
    src/ub_oracle/completeness.py. -/
def guaranteedClassKeys : List String :=
  ["signed_overflow", "shift_oob", "div_by_zero", "intmin_div_neg1"]

/-- Classes explicitly outside that fragment: the implementation may abstain, or
    report soundly when another oracle applies, but makes no completeness claim. -/
def mayAbstainClassKeys : List String :=
  ["array_oob", "strict_aliasing", "fp_contraction"]

/-- True exactly for classes complete on their declared finite fragment. -/
def inCompletenessFragment : BoundaryClass -> Bool
  | BoundaryClass.signedOverflow => true
  | BoundaryClass.shiftOutOfRange => true
  | BoundaryClass.divisionByZero => true
  | BoundaryClass.intMinDivNeg1 => true
  | BoundaryClass.arrayOob => false
  | BoundaryClass.strictAliasing => false
  | BoundaryClass.fpContraction => false

/-- True exactly for classes outside the completeness fragment. -/
def mayAbstainClass : BoundaryClass -> Bool
  | BoundaryClass.signedOverflow => false
  | BoundaryClass.shiftOutOfRange => false
  | BoundaryClass.divisionByZero => false
  | BoundaryClass.intMinDivNeg1 => false
  | BoundaryClass.arrayOob => true
  | BoundaryClass.strictAliasing => true
  | BoundaryClass.fpContraction => true

/-- Public predicate: the tool guarantees to catch the class on the class's
    declared finite fragment. -/
def guaranteedToCatchOnDeclaredRange (c : BoundaryClass) : Bool :=
  inCompletenessFragment c

/-- A compact boundary claim, used by the totality/disjointness theorem below. -/
inductive BoundaryClaim where
  | completeOnDeclaredFragment
  | soundButMayAbstain
deriving DecidableEq, Repr

def boundaryClaim (c : BoundaryClass) : BoundaryClaim :=
  if inCompletenessFragment c then
    BoundaryClaim.completeOnDeclaredFragment
  else
    BoundaryClaim.soundButMayAbstain

/-- Every published class is classified on one side of the boundary. -/
theorem completeness_boundary_total (c : BoundaryClass) :
    guaranteedToCatchOnDeclaredRange c = true ∨ mayAbstainClass c = true := by
  cases c <;> simp [guaranteedToCatchOnDeclaredRange, inCompletenessFragment,
    mayAbstainClass]

/-- No published class is both guaranteed-complete and may-abstain. -/
theorem completeness_boundary_disjoint (c : BoundaryClass) :
    ¬ (guaranteedToCatchOnDeclaredRange c = true ∧ mayAbstainClass c = true) := by
  cases c <;> simp [guaranteedToCatchOnDeclaredRange, inCompletenessFragment,
    mayAbstainClass]

/-- The combined boundary theorem: every class is exactly one of
    complete-on-declared-fragment or sound-but-may-abstain. -/
theorem mechanized_completeness_boundary (c : BoundaryClass) :
    (guaranteedToCatchOnDeclaredRange c = true ∧ mayAbstainClass c = false) ∨
    (guaranteedToCatchOnDeclaredRange c = false ∧ mayAbstainClass c = true) := by
  cases c <;> simp [guaranteedToCatchOnDeclaredRange, inCompletenessFragment,
    mayAbstainClass]

/-- Out-of-fragment classes carry no completeness guarantee. -/
theorem out_of_fragment_abstains_only (c : BoundaryClass) :
    mayAbstainClass c = true -> guaranteedToCatchOnDeclaredRange c = false := by
  intro h
  cases c <;> simp [guaranteedToCatchOnDeclaredRange, inCompletenessFragment,
    mayAbstainClass] at h ⊢

/-- In-fragment classes inherit the recorded-observable decision theorem.  The
    concrete per-class finite-range evidence remains the executable brute-force
    check in src/ub_oracle/completeness.py. -/
theorem complete_fragment_decides_recorded_observation
    (c : BoundaryClass) (o : Observation) :
    inCompletenessFragment c = true ->
      (productViolated o = true ↔ isUBDivergence o) := by
  intro _
  exact oracle_decides o

/-- The Boolean classifier and the compact claim agree. -/
theorem boundary_claim_matches_predicates (c : BoundaryClass) :
    (boundaryClaim c = BoundaryClaim.completeOnDeclaredFragment ↔
      guaranteedToCatchOnDeclaredRange c = true) ∧
    (boundaryClaim c = BoundaryClaim.soundButMayAbstain ↔
      mayAbstainClass c = true) := by
  cases c <;> simp [boundaryClaim, guaranteedToCatchOnDeclaredRange,
    inCompletenessFragment, mayAbstainClass]

example :
    boundaryClaim BoundaryClass.signedOverflow =
      BoundaryClaim.completeOnDeclaredFragment := by decide

example :
    boundaryClaim BoundaryClass.strictAliasing =
      BoundaryClaim.soundButMayAbstain := by decide

end CrossLangVerifier
