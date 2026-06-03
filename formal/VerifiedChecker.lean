import ProductSoundness

namespace CrossLangVerifier

inductive Verdict where
  | divergent
  | candidate
  | noDivergenceFound
  | unknown
  | notCovered
deriving DecidableEq, Repr

structure Claim where
  verdict       : Verdict
  ubReached     : Bool
  targetDefined : Bool
  consequence   : Bool
deriving DecidableEq, Repr

def Claim.observation (c : Claim) : Observation :=
  { ubReached := c.ubReached
    tgtDefined := c.targetDefined
    consequence := c.consequence }

def verifiesPositiveClaim (c : Claim) : Bool :=
  match c.verdict with
  | Verdict.divergent => productViolated c.observation
  | _ => true

theorem divergent_claim_sound (ub targetDefined consequence : Bool) :
    productViolated
      ({ ubReached := ub
         tgtDefined := targetDefined
         consequence := consequence } : Observation) = true →
      isUBDivergence
        ({ ubReached := ub
           tgtDefined := targetDefined
           consequence := consequence } : Observation) := by
  exact oracle_sound _

theorem accepted_divergent_claim_sound
    (ub targetDefined consequence : Bool) :
    verifiesPositiveClaim
      { verdict := Verdict.divergent
        ubReached := ub
        targetDefined := targetDefined
        consequence := consequence } = true →
      isUBDivergence
        ({ ubReached := ub
           tgtDefined := targetDefined
           consequence := consequence } : Observation) := by
  intro haccepted
  simp [verifiesPositiveClaim, Claim.observation] at haccepted
  exact oracle_sound _ haccepted

/-- The certificate-level predicate modeled by the executable boundary.

    The Python replay layer checks the concrete JSON schema, canonical hash
    binding, and source-UB scope.  The extracted checker then validates the
    recorded-observable claim.  This predicate names that combined boundary in
    Lean without pretending that SHA-256 or JSON parsing are mechanized here. -/
structure ProofCertificate where
  claim                : Claim
  counterexampleBound  : Bool
  certificateHashBound : Bool
  sourceUBScope        : Bool
deriving DecidableEq, Repr

def ProofCertificate.observation (cert : ProofCertificate) : Observation :=
  cert.claim.observation

def verifiesProofCertificate (cert : ProofCertificate) : Bool :=
  cert.counterexampleBound &&
    cert.certificateHashBound &&
    cert.sourceUBScope &&
    match cert.claim.verdict with
    | Verdict.divergent => verifiesPositiveClaim cert.claim
    | _ => false

theorem proof_carrying_counterexample_sound (cert : ProofCertificate) :
    verifiesProofCertificate cert = true →
      isUBDivergence cert.observation := by
  cases cert with
  | mk claim counterexampleBound certificateHashBound sourceUBScope =>
    cases claim with
    | mk verdict ubReached targetDefined consequence =>
      cases verdict <;>
        simp [verifiesProofCertificate, ProofCertificate.observation,
          verifiesPositiveClaim, Claim.observation, productViolated, R,
          isUBDivergence]
      intro _counterexampleBound _certificateHashBound _sourceUBScope
        hubReached htgtDefined hconsequence
      exact ⟨hubReached, htgtDefined, hconsequence⟩

def parseBool? (s : String) : Option Bool :=
  if s == "true" then some true
  else if s == "false" then some false
  else none

def parseVerdict? (s : String) : Option Verdict :=
  if s == "divergent" then some Verdict.divergent
  else if s == "candidate" then some Verdict.candidate
  else if s == "no_divergence_found" then some Verdict.noDivergenceFound
  else if s == "unknown" then some Verdict.unknown
  else if s == "not_covered" then some Verdict.notCovered
  else none

def verdictText : Verdict → String
  | Verdict.divergent => "divergent"
  | Verdict.candidate => "candidate"
  | Verdict.noDivergenceFound => "no_divergence_found"
  | Verdict.unknown => "unknown"
  | Verdict.notCovered => "not_covered"

def boolText : Bool → String
  | true => "true"
  | false => "false"

structure RawArgs where
  verdict?       : Option String := none
  ub?            : Option String := none
  targetDefined? : Option String := none
  consequence?   : Option String := none
deriving Repr

def parseRawArgs : List String → RawArgs → Except String RawArgs
  | [], raw => Except.ok raw
  | "--verdict" :: v :: rest, raw =>
      parseRawArgs rest { raw with verdict? := some v }
  | "--ub" :: v :: rest, raw =>
      parseRawArgs rest { raw with ub? := some v }
  | "--target-defined" :: v :: rest, raw =>
      parseRawArgs rest { raw with targetDefined? := some v }
  | "--consequence" :: v :: rest, raw =>
      parseRawArgs rest { raw with consequence? := some v }
  | flag :: _, _ =>
      Except.error s!"unknown or incomplete argument: {flag}"

def requiredString (name : String) : Option String → Except String String
  | some v => Except.ok v
  | none => Except.error s!"missing required argument: {name}"

def requiredBool (name : String) (value? : Option String) : Except String Bool := do
  let raw ← requiredString name value?
  match parseBool? raw with
  | some b => Except.ok b
  | none => Except.error s!"{name} must be true or false, got: {raw}"

def requiredVerdict (value? : Option String) : Except String Verdict := do
  let raw ← requiredString "--verdict" value?
  match parseVerdict? raw with
  | some v => Except.ok v
  | none => Except.error s!"unknown verdict: {raw}"

def parseClaim (args : List String) : Except String Claim := do
  let raw ← parseRawArgs args {}
  let verdict ← requiredVerdict raw.verdict?
  let ub ← requiredBool "--ub" raw.ub?
  let targetDefined ← requiredBool "--target-defined" raw.targetDefined?
  let consequence ← requiredBool "--consequence" raw.consequence?
  Except.ok
    { verdict := verdict
      ubReached := ub
      targetDefined := targetDefined
      consequence := consequence }

def jsonLine (claim : Claim) (accepted : Bool) : String :=
  "{"
  ++ "\"accepted\":" ++ boolText accepted ++ ","
  ++ "\"verdict\":\"" ++ verdictText claim.verdict ++ "\","
  ++ "\"ub_reached\":" ++ boolText claim.ubReached ++ ","
  ++ "\"target_defined\":" ++ boolText claim.targetDefined ++ ","
  ++ "\"consequence\":" ++ boolText claim.consequence ++ ","
  ++ "\"product_violated\":" ++ boolText (productViolated claim.observation) ++ ","
  ++ "\"kernel_theorem\":\"oracle_sound\","
  ++ "\"scope\":\"final source-UB positive-claim inference over trusted run facts\""
  ++ "}"

end CrossLangVerifier

open CrossLangVerifier

def main (args : List String) : IO UInt32 := do
  match parseClaim args with
  | Except.error err =>
      IO.eprintln s!"verified-checker: {err}"
      return 2
  | Except.ok claim =>
      let accepted := verifiesPositiveClaim claim
      IO.println (jsonLine claim accepted)
      if accepted then
        return 0
      else
        return 1
