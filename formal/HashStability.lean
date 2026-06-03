/-
  Step 138: mechanized hash-stability lemma for the proof-certificate verdict
  layer.

  The executable certificate hash is SHA-256 over a typed, length-prefixed
  preimage emitted by src/ub_oracle/replay.py.  This file deliberately proves the
  part that can be true by construction: the canonical verdict-layer preimage is
  injective.  It does not pretend SHA-256 itself is collision-free; rather, two
  distinct verdict payloads can share a certificate hash only through a SHA-256
  collision after this injective canonicalization step.
-/

namespace CrossLangVerifier.HashStability

/-- The fixed scalar verdict layer bound into a proof certificate.  Rich
    counterexample transcripts are hashed separately as JSON; this is the
    source-UB positive-claim layer that the verified checker consumes. -/
structure VerdictHashPayload where
  schemaVersion     : Nat
  verdict           : String
  ubReached         : Bool
  targetDefined     : Bool
  consequence       : Bool
  kernelTheorem     : String
  checkerScope      : String
  issuer            : String
  counterexampleHash : String
deriving DecidableEq, Repr

/-- Typed canonical atoms.  The Python encoder serializes this fixed sequence
    with explicit tags and UTF-8 byte lengths; the proof only needs the typed
    token layer to show that field boundaries cannot be ambiguous. -/
inductive CanonToken where
  | natField    : String -> Nat -> CanonToken
  | stringField : String -> String -> CanonToken
  | boolField   : String -> Bool -> CanonToken
deriving DecidableEq, Repr

def verdictCanonicalTokens (p : VerdictHashPayload) : List CanonToken := [
  CanonToken.natField "schema_version" p.schemaVersion,
  CanonToken.stringField "verdict" p.verdict,
  CanonToken.boolField "observation.ub_reached" p.ubReached,
  CanonToken.boolField "observation.target_defined" p.targetDefined,
  CanonToken.boolField "observation.consequence" p.consequence,
  CanonToken.stringField "kernel_theorem" p.kernelTheorem,
  CanonToken.stringField "checker_scope" p.checkerScope,
  CanonToken.stringField "issuer" p.issuer,
  CanonToken.stringField "counterexample_hash" p.counterexampleHash
]

/-- Main Step-138 lemma: the verdict-layer canonical preimage is injective.

    Therefore canonicalization itself cannot merge two different verdict
    payloads; any equality of the final fixed-width SHA-256 strings would have to
    be a collision in SHA-256, not a boundary ambiguity in our encoding. -/
theorem verdict_canonicalization_injective
    {a b : VerdictHashPayload} :
    verdictCanonicalTokens a = verdictCanonicalTokens b -> a = b := by
  cases a
  cases b
  simp [verdictCanonicalTokens]

/-- A consequence bit flip changes the canonical preimage. -/
theorem verdict_canonicalization_distinguishes_consequence
    (p : VerdictHashPayload) :
    verdictCanonicalTokens { p with consequence := !p.consequence } ≠
      verdictCanonicalTokens p := by
  cases p with
  | mk schemaVersion verdict ubReached targetDefined consequence
      kernelTheorem checkerScope issuer counterexampleHash =>
    cases consequence <;> simp [verdictCanonicalTokens]

/-- The observation triple is recovered from equal canonical preimages. -/
theorem verdict_canonicalization_preserves_observation
    {a b : VerdictHashPayload} :
    verdictCanonicalTokens a = verdictCanonicalTokens b ->
      a.ubReached = b.ubReached
        /\ a.targetDefined = b.targetDefined
        /\ a.consequence = b.consequence := by
  intro h
  have hp := verdict_canonicalization_injective h
  cases hp
  exact ⟨rfl, rfl, rfl⟩

end CrossLangVerifier.HashStability
