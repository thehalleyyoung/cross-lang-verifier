import Lake
open Lake DSL

package «cross-lang-verifier-formal» where

lean_lib ProductSoundness where
  roots := #[`ProductSoundness]

lean_lib CompletenessBoundary where
  roots := #[`CompletenessBoundary]

lean_lib SPIContract where
  roots := #[`SPIContract]

lean_lib HashStability where
  roots := #[`HashStability]

lean_exe «verified-checker» where
  root := `VerifiedChecker
