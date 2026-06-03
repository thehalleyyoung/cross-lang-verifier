import Lake
open Lake DSL

package «cross-lang-verifier-formal» where

lean_lib ProductSoundness where
  roots := #[`ProductSoundness]

lean_lib CompletenessBoundary where
  roots := #[`CompletenessBoundary]

lean_exe «verified-checker» where
  root := `VerifiedChecker
