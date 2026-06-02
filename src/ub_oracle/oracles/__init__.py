"""Concrete divergence-class oracle plugins.

Importing this package registers all built-in oracles into the plugin registry.
"""

from . import signed_overflow  # noqa: F401  (registers SignedOverflowOracle)
from . import integer_ub  # noqa: F401  (registers shift / div-by-zero / INT_MIN-div-neg1)
from . import memory_shape  # noqa: F401  (registers array-OOB / strict-aliasing)
from . import uninit_read  # noqa: F401  (registers uninitialized-read / definedness)
from . import floating_point  # noqa: F401  (registers FP-contraction)
from . import vla_bound  # noqa: F401  (registers VLA-bound C->Rust and C->Go)
from . import float_cast  # noqa: F401  (registers float->int overflow C->Rust and C->Go)
from . import fast_math  # noqa: F401  (registers -ffast-math reassociation C->Rust and C->Go)
from . import restrict_alias  # noqa: F401  (registers restrict-violation C->Rust and C->Go)
from . import pointer_provenance  # noqa: F401  (registers pointer-provenance C->Rust and C->Go)
from . import target_pairs  # noqa: F401  (registers C->Go and C->Swift pairs)
from . import c_to_cpp  # noqa: F401  (registers C->C++ defined-subset pair, step 117)
from . import go_to_rust  # noqa: F401  (registers Go->Rust defined-but-different pair, step 120)
from . import bitfield_layout  # noqa: F401  (registers bit-field layout C->Rust and C->Go, step 112)
from . import enum_repr  # noqa: F401  (registers out-of-range enum C->Rust and C->Go, step 108)
from . import memcpy_overlap  # noqa: F401  (registers memcpy-overlap C->Rust and C->Go, step 103)
from . import sequence_point  # noqa: F401  (registers unsequenced-modification C->Rust, step 105)
from . import longjmp_vla  # noqa: F401  (registers longjmp-over-exited-VLA, step 111)
from . import c_to_go  # noqa: F401  (back-compat shim exposing the Go oracles)

__all__ = ["signed_overflow", "integer_ub", "memory_shape", "uninit_read",
           "floating_point", "vla_bound", "target_pairs", "c_to_go", "c_to_cpp",
           "go_to_rust", "bitfield_layout", "enum_repr", "memcpy_overlap",
           "sequence_point", "longjmp_vla"]
