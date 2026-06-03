(*
  Independent Coq cross-check for the core product-program soundness theorem
  (100_STEPS step 130).

  This file intentionally does not import the Lean development.  It re-proves the
  central recorded-observable lemma in Coq's kernel: if the product assertion
  R = not (source-UB /\ target-defined /\ behaviours-differ) is violated, then
  the recorded observation is a genuine UB-rooted divergence.  The product-run
  witness theorem also proves the counterexample constructor preserves the
  concrete source/target/input payload and derives its observation from raw run
  facts.
*)

From Coq Require Import Bool.Bool.
From Coq Require Import Strings.String.
From Coq Require Import ZArith.ZArith.

Open Scope string_scope.
Open Scope Z_scope.

Record Observation := {
  ubReached : bool;
  tgtDefined : bool;
  consequence : bool
}.

Definition R (o : Observation) : bool :=
  negb (ubReached o && tgtDefined o && consequence o).

Definition productViolated (o : Observation) : bool :=
  negb (R o).

Definition isUBDivergence (o : Observation) : Prop :=
  ubReached o = true /\ tgtDefined o = true /\ consequence o = true.

Definition equivalent (o : Observation) : Prop :=
  consequence o = false.

Theorem oracle_sound_coq :
  forall o, productViolated o = true -> isUBDivergence o.
Proof.
  intros [ub tgt diff] H.
  unfold productViolated, R in H; simpl in H.
  destruct ub, tgt, diff; simpl in H; try discriminate;
    repeat split; reflexivity.
Qed.

Theorem oracle_complete_rel_coq :
  forall o, isUBDivergence o -> productViolated o = true.
Proof.
  intros [ub tgt diff] [Hub [Htgt Hdiff]].
  subst; reflexivity.
Qed.

Theorem oracle_decides_coq :
  forall o, productViolated o = true <-> isUBDivergence o.
Proof.
  intro o; split.
  - apply oracle_sound_coq.
  - apply oracle_complete_rel_coq.
Qed.

Theorem equivalence_never_reported_coq :
  forall o, equivalent o -> productViolated o = false.
Proof.
  intros [ub tgt diff] H.
  unfold equivalent in H; simpl in H; subst.
  destruct ub, tgt; reflexivity.
Qed.

Theorem report_implies_ub_coq :
  forall o, productViolated o = true -> ubReached o = true.
Proof.
  intros o H.
  destruct (oracle_sound_coq o H) as [Hub _].
  exact Hub.
Qed.

Record TargetPack := {
  packName : string;
  definedReturn : Z -> bool
}.

Definition observe
    (pack : TargetPack)
    (ub : bool)
    (targetReturn : Z)
    (behavioursDiffer : bool) : Observation :=
  {| ubReached := ub;
     tgtDefined := definedReturn pack targetReturn;
     consequence := behavioursDiffer |}.

Theorem pack_oracle_sound_coq :
  forall pack ub rc diff,
    productViolated (observe pack ub rc diff) = true ->
    isUBDivergence (observe pack ub rc diff).
Proof.
  intros; apply oracle_sound_coq; assumption.
Qed.

Definition rustDefinedReturn (rc : Z) : bool :=
  Z.eqb rc 0 || Z.eqb rc 101.

Definition RustPack : TargetPack :=
  {| packName := "rust"; definedReturn := rustDefinedReturn |}.

Theorem rust_oracle_sound_coq :
  forall ub rc diff,
    productViolated (observe RustPack ub rc diff) = true ->
    isUBDivergence (observe RustPack ub rc diff).
Proof.
  intros; apply pack_oracle_sound_coq; assumption.
Qed.

Example rust_div_by_zero_positive_coq :
  productViolated (observe RustPack true 101 true) = true /\
  isUBDivergence (observe RustPack true 101 true).
Proof.
  split.
  - reflexivity.
  - apply rust_oracle_sound_coq; reflexivity.
Qed.

Example rust_safe_negative_coq :
  productViolated (observe RustPack false 0 false) = false.
Proof. reflexivity. Qed.

Record ProductRun := {
  runSourceId : string;
  runTargetId : string;
  runInput : string;
  runPack : TargetPack;
  runUbReached : bool;
  runTargetReturn : Z;
  runBehavioursDiffer : bool
}.

Record ProductCounterexample := {
  cexSourceId : string;
  cexTargetId : string;
  cexInput : string;
  cexObservation : Observation
}.

Definition productRunObservation (r : ProductRun) : Observation :=
  observe (runPack r) (runUbReached r) (runTargetReturn r)
    (runBehavioursDiffer r).

Definition productRunToCounterexample
    (r : ProductRun) : option ProductCounterexample :=
  if productViolated (productRunObservation r)
  then Some {| cexSourceId := runSourceId r;
               cexTargetId := runTargetId r;
               cexInput := runInput r;
               cexObservation := productRunObservation r |}
  else None.

Theorem product_program_preserves_divergence_witness_coq :
  forall r cex,
    productRunToCounterexample r = Some cex ->
    cexSourceId cex = runSourceId r /\
    cexTargetId cex = runTargetId r /\
    cexInput cex = runInput r /\
    cexObservation cex = productRunObservation r /\
    isUBDivergence (cexObservation cex).
Proof.
  intros r cex H.
  unfold productRunToCounterexample in H.
  destruct (productViolated (productRunObservation r)) eqn:Hviol.
  - inversion H; subst; simpl.
    repeat split; try reflexivity.
    apply oracle_sound_coq; assumption.
  - discriminate.
Qed.

Theorem product_program_emits_witness_iff_product_violated_coq :
  forall r,
    (exists cex, productRunToCounterexample r = Some cex) <->
    productViolated (productRunObservation r) = true.
Proof.
  intro r; split.
  - intros [cex H].
    destruct (product_program_preserves_divergence_witness_coq r cex H)
      as [_ [_ [_ [Hobs Hdiv]]]].
    rewrite Hobs in Hdiv.
    apply oracle_complete_rel_coq; assumption.
  - intro Hviol.
    unfold productRunToCounterexample.
    rewrite Hviol.
    eexists; reflexivity.
Qed.

Theorem product_program_witness_iff_divergence_coq :
  forall r,
    (exists cex, productRunToCounterexample r = Some cex) <->
    isUBDivergence (productRunObservation r).
Proof.
  intro r; split.
  - intro Hemits.
    apply oracle_sound_coq.
    apply product_program_emits_witness_iff_product_violated_coq.
    exact Hemits.
  - intro Hdiv.
    apply product_program_emits_witness_iff_product_violated_coq.
    apply oracle_complete_rel_coq.
    exact Hdiv.
Qed.

Example product_program_positive_preserves_payload_coq :
  exists cex,
    productRunToCounterexample
      {| runSourceId := "c:div";
         runTargetId := "rust:div";
         runInput := "b=0";
         runPack := RustPack;
         runUbReached := true;
         runTargetReturn := 101;
         runBehavioursDiffer := true |} = Some cex /\
    cexSourceId cex = "c:div" /\
    cexTargetId cex = "rust:div" /\
    cexInput cex = "b=0" /\
    isUBDivergence (cexObservation cex).
Proof.
  simpl.
  eexists; repeat split; reflexivity.
Qed.
