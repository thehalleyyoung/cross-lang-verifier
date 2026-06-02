# Responsible-disclosure template

Use this template to report a confirmed cross-language translation divergence to a project's maintainers. Fill every section; attach the reproduction bundle emitted by `ub_oracle.disclosure.reproduce_disclosure`.

- **Advisory ID:** `CLV-…`
- **Title:** one line.
- **Affected pattern / provenance:** the real-world idiom and where it came from.
- **Affected pair:** `C -> <target>`.
- **Summary:** what diverges and why (root the bug in a specific C undefined behaviour).
- **Impact:** correctness / security consequence on a real input.
- **Proof of concept:** the witnessing input + the attached runnable reproduction bundle.
- **Defined (safe) input:** an input on which both sides agree (shows the bug is input-specific, not a wholesale mistranslation).
- **Remediation:** the concrete fix that makes both sides agree.
- **Disclosure timeline:** report date, maintainer ack, fix, public date (coordinated).
