import ProductSoundness

/-
  Formal SPI contract (100_STEPS step 136).

  ProductSoundness.lean proves the recorded-observable soundness theorem used by
  the verifier.  This file states the *interfaces* that target-semantics packs
  and divergence-oracle plugins must satisfy as Lean typeclasses.  A conforming
  target pack must prove that its executable "defined outcome" predicate is
  exactly the declared return-code data, that canonicalization only accepts
  declared defined outcomes, and that any reported product-program violation
  routes through ProductSoundness.pack_oracle_sound.  A conforming plugin must
  expose the same method surface as src/ub_oracle/plugin.py and prove that any
  found/confirmed witness is a product-program violation and therefore a
  genuine divergence in the recorded-observable abstraction.

  Honest scope: the Lean typeclasses do not verify compiler command lines,
  filesystem staging, or richer runtime predicates such as WebAssembly trap
  stderr classification.  Those obligations remain executable evidence in
  src/ub_oracle/pack_conformance.py.  The companion Python checker
  src/ub_oracle/spi_contract.py ties this file to that executable suite: every
  obligation key and every registered target pack must be present on both sides.
-/

namespace CrossLangVerifier

/-- Canonical target outcomes used by the formal return-code layer.  Runtime
    predicates with richer evidence (for example wasm trap stderr) are explicitly
    handled by the executable pack-conformance suite. -/
inductive CanonicalOutcome where
  | returnedValue
  | deterministicAbort
deriving DecidableEq, Repr

/-- Canonicalize a process return code using only the declared finite set of
    target-defined return codes. -/
def canonicalizeFromCodes (codes : List Int) (rc : Int) : Option CanonicalOutcome :=
  if codes.contains rc = true then
    some (if rc == 0 then CanonicalOutcome.returnedValue else
      CanonicalOutcome.deterministicAbort)
  else
    none

/-- Canonicalization is sound with respect to the declared finite data:
    it cannot accept a return code outside the pack's declared defined set. -/
theorem canonicalizeFromCodes_sound
    (codes : List Int) (rc : Int) (outcome : CanonicalOutcome) :
    canonicalizeFromCodes codes rc = some outcome -> codes.contains rc = true := by
  intro h
  unfold canonicalizeFromCodes at h
  by_cases hc : codes.contains rc = true
  · exact hc
  · simp [hc] at h
    exact (by simpa using h.1)

/-- The finite model of a target-semantics pack used by the formal contract.
    The `pack` field is the ProductSoundness target pack; the remaining fields
    are the data obligations mirrored by the Python registry. -/
structure TargetPackModel where
  key : String
  pack : TargetPack
  declaredCodes : List Int
  canonicalize : Int -> Option CanonicalOutcome
  requiresRuntimeOutcomePredicate : Bool

/-- Typeclass obligations for a target-semantics pack.  A pack instance cannot
    discharge this contract unless its ProductSoundness predicate agrees with
    its declared data and its product-program soundness proof factors through
    the already-proved pack_oracle_sound theorem. -/
class TargetPackSPI (m : TargetPackModel) where
  nameMatchesPack :
    m.pack.name = m.key
  definedReturnMatchesData :
    ∀ rc, m.pack.definedReturn rc = m.declaredCodes.contains rc
  canonicalizationSound :
    ∀ rc outcome, m.canonicalize rc = some outcome ->
      m.declaredCodes.contains rc = true
  zeroIsDefined :
    m.pack.definedReturn 0 = true
  productSound :
    ∀ ub rc diff,
      productViolated (observe m.pack ub rc diff) = true ->
        isUBDivergence (observe m.pack ub rc diff)

/-- The executable obligation names in src/ub_oracle/pack_conformance.py.  The
    Python bridge checks this list byte-for-byte against that module's
    OBLIGATION_KEYS. -/
def targetPackObligationKeys : List String := [
  "name_matches_registry",
  "suffix_is_a_dot_ext",
  "compilers_declared",
  "defined_rc_well_formed",
  "predicate_is_total",
  "run_predicate_consistent",
  "compile_argv_wires_io",
  "compile_env_is_a_dict",
  "runner_argv_wires_artifact",
  "resolutions_are_real"
]

theorem target_pack_obligation_keys_cover_executable_suite :
    targetPackObligationKeys = [
      "name_matches_registry",
      "suffix_is_a_dot_ext",
      "compilers_declared",
      "defined_rc_well_formed",
      "predicate_is_total",
      "run_predicate_consistent",
      "compile_argv_wires_io",
      "compile_env_is_a_dict",
      "runner_argv_wires_artifact",
      "resolutions_are_real"
    ] := rfl

def codeBackedPack (name : String) (codes : List Int) : TargetPack where
  name := name
  definedReturn := fun rc => codes.contains rc

def codeBackedModel
    (name : String) (codes : List Int) (runtimePredicate : Bool) :
    TargetPackModel where
  key := name
  pack := codeBackedPack name codes
  declaredCodes := codes
  canonicalize := canonicalizeFromCodes codes
  requiresRuntimeOutcomePredicate := runtimePredicate

def codeBackedSPI
    (name : String) (codes : List Int) (runtimePredicate : Bool)
    (hzero : codes.contains (0 : Int) = true) :
    TargetPackSPI (codeBackedModel name codes runtimePredicate) where
  nameMatchesPack := rfl
  definedReturnMatchesData := by
    intro rc
    rfl
  canonicalizationSound := by
    intro rc outcome h
    exact canonicalizeFromCodes_sound codes rc outcome h
  zeroIsDefined := by
    exact hzero
  productSound := by
    intro ub rc diff h
    exact pack_oracle_sound _ ub rc diff h

/-! ### Concrete target-pack instances mirrored from src/ub_oracle/target_semantics.py -/

def rustPackModel : TargetPackModel :=
  codeBackedModel "rust" [0, 101] false

instance rustTargetPackSPI : TargetPackSPI rustPackModel :=
  codeBackedSPI "rust" [0, 101] false (by decide)

def goPackModel : TargetPackModel :=
  codeBackedModel "go" [0, 2] false

instance goTargetPackSPI : TargetPackSPI goPackModel :=
  codeBackedSPI "go" [0, 2] false (by decide)

def swiftPackModel : TargetPackModel :=
  codeBackedModel "swift" [0, -5] false

instance swiftTargetPackSPI : TargetPackSPI swiftPackModel :=
  codeBackedSPI "swift" [0, -5] false (by decide)

def cppPackModel : TargetPackModel :=
  codeBackedModel "cpp" [0] false

instance cppTargetPackSPI : TargetPackSPI cppPackModel :=
  codeBackedSPI "cpp" [0] false (by decide)

def ocamlPackModel : TargetPackModel :=
  codeBackedModel "ocaml" [0, 2] false

instance ocamlTargetPackSPI : TargetPackSPI ocamlPackModel :=
  codeBackedSPI "ocaml" [0, 2] false (by decide)

def zigPackModel : TargetPackModel :=
  codeBackedModel "zig" [0, -6] false

instance zigTargetPackSPI : TargetPackSPI zigPackModel :=
  codeBackedSPI "zig" [0, -6] false (by decide)

/-- WebAssembly has an executable runtime-stderr trap predicate in Python; the
    formal return-code layer still proves the rc=0 value outcome and records that
    the richer predicate must be checked by the executable conformance suite. -/
def wasmPackModel : TargetPackModel :=
  codeBackedModel "wasm" [0] true

instance wasmTargetPackSPI : TargetPackSPI wasmPackModel :=
  codeBackedSPI "wasm" [0] true (by decide)

example : TargetPackSPI rustPackModel := inferInstance
example : TargetPackSPI goPackModel := inferInstance
example : TargetPackSPI swiftPackModel := inferInstance
example : TargetPackSPI cppPackModel := inferInstance
example : TargetPackSPI ocamlPackModel := inferInstance
example : TargetPackSPI zigPackModel := inferInstance
example : TargetPackSPI wasmPackModel := inferInstance

/-- The method surface of src/ub_oracle/plugin.py:DivergenceOracle.  `confirm` is
    a concrete method rather than an abstractmethod, but it is still part of the
    plugin SPI and therefore appears in the formal contract. -/
def oraclePluginMethodKeys : List String :=
  ["applies_to", "find_divergence", "confirm"]

theorem oracle_plugin_method_keys_cover_python_spi :
    oraclePluginMethodKeys = ["applies_to", "find_divergence", "confirm"] := rfl

structure OraclePluginModel where
  key : String
  methodKeys : List String
  applies : Observation -> Bool
  findDivergence : Observation -> Option Observation
  confirm : Observation -> Bool

/-- Typeclass obligations for a divergence-oracle plugin.  They match the real
    Python methods: `applies_to`, `find_divergence`, and `confirm`. -/
class OraclePluginSPI (p : OraclePluginModel) where
  methodKeysMatchPython :
    p.methodKeys = oraclePluginMethodKeys
  silentWhenNotApplicable :
    ∀ input, p.applies input = false -> p.findDivergence input = none
  finderSound :
    ∀ input witness, p.findDivergence input = some witness ->
      productViolated witness = true
  confirmationSound :
    ∀ witness, p.confirm witness = true ->
      productViolated witness = true -> isUBDivergence witness

/-- The core recorded-observable oracle used to prove that the plugin SPI is not
    vacuous: it reports exactly product-program violations. -/
def recordedObservableOracle : OraclePluginModel where
  key := "recorded_observable"
  methodKeys := oraclePluginMethodKeys
  applies := productViolated
  findDivergence := fun input =>
    if productViolated input = true then some input else none
  confirm := productViolated

instance recordedObservableOracleSPI :
    OraclePluginSPI recordedObservableOracle where
  methodKeysMatchPython := rfl
  silentWhenNotApplicable := by
    intro input h
    change productViolated input = false at h
    simp [recordedObservableOracle, h]
  finderSound := by
    intro input witness h
    change (if productViolated input = true then some input else none) =
      some witness at h
    by_cases hv : productViolated input = true
    · simp [hv] at h
      cases h
      exact hv
    · simp [hv] at h
  confirmationSound := by
    intro witness _ hproduct
    exact oracle_sound witness hproduct

theorem recorded_observable_oracle_sound
    (input witness : Observation) :
    recordedObservableOracle.findDivergence input = some witness ->
      isUBDivergence witness := by
  intro h
  have hv : productViolated witness = true :=
    OraclePluginSPI.finderSound input witness h
  exact oracle_sound witness hv

example :
    recordedObservableOracle.findDivergence (observe rustPackModel.pack true 101 true) =
      some (observe rustPackModel.pack true 101 true) := by decide

example :
    recordedObservableOracle.findDivergence (observe rustPackModel.pack false 0 false) =
      none := by decide

end CrossLangVerifier
