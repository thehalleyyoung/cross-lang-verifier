# The divergence zoo

*Auto-generated from the live corpora — do not edit by hand; run `python -m ub_oracle.divergence_zoo`.*

A machine-readable, indexed catalogue of the cross-language divergence patterns this tool catches, **indexed by class and language pair**. Every divergent exhibit carries a concrete **witnessing input** and is **re-confirmed live** by `confirm_zoo()` (the oracle must still flag the divergence on the witness and stay silent on the safe input).

*content hash: `009f00d99eb82f44` — 14 divergent exhibits across 3 classes.*

## Index — class × pair

| divergence class | language pair | exhibits |
|------------------|---------------|----------|
| `div_by_zero` | `c->go` | `idio:rate-divide:go`, `multi:rate:go` |
| `div_by_zero` | `c->rust` | `idio:rate-divide:rust`, `multi:rate:rust` |
| `div_by_zero` | `c->swift` | `multi:rate:swift` |
| `oversized_shift` | `c->go` | `multi:bitfield:go` |
| `oversized_shift` | `c->rust` | `idio:bitfield-shift:rust`, `multi:bitfield:rust` |
| `oversized_shift` | `c->swift` | `multi:bitfield:swift` |
| `signed_overflow` | `c->go` | `idio:midpoint-overflow:go`, `multi:midpoint:go` |
| `signed_overflow` | `c->rust` | `idio:midpoint-overflow:rust`, `multi:midpoint:rust` |
| `signed_overflow` | `c->swift` | `multi:midpoint:swift` |

## Exhibits

### `idio:bitfield-shift:rust` — oversized_shift (c->rust)

*Mirrors:* bit-field / flag extraction `v << w` (as in packed-struct decoders); a width >= 32 is out-of-range UB in C, but Rust's `wrapping_shl` is defined.. *Witness:* `['1', '40']` (safe: `['1', '3']`).

```c
#include <stdio.h>
#include <stdlib.h>
static int field(int v,int w){return v<<w;}
int main(int argc,char**argv){int v=atoi(argv[1]);int w=atoi(argv[2]);printf("%d\n",field(v,w));return 0;}
```

```rust
fn field(v:i32,w:u32)->i32{ v.wrapping_shl(w) }
fn main(){
  let v: i32 = std::env::args().nth(1).unwrap().parse().unwrap();
  let w: u32 = std::env::args().nth(2).unwrap().parse().unwrap();
  println!("{}", field(v,w));
}
```

### `idio:midpoint-overflow:go` — signed_overflow (c->go)

*Mirrors:* binary-search / merge midpoint `(lo+hi)/2` (the JDK / NIST-famous signed-overflow bug); idiomatic ports use wrapping (Rust) or 64-bit `int` (Go), so they are defined where C is UB.. *Witness:* `['2000000000', '2000000000']` (safe: `['10', '20']`).

```c
#include <stdio.h>
#include <stdlib.h>
static int midpoint(int lo,int hi){return (lo+hi)/2;}
int main(int argc,char**argv){int lo=atoi(argv[1]);int hi=atoi(argv[2]);printf("%d\n",midpoint(lo,hi));return 0;}
```

```go
package main
import ("fmt";"os";"strconv")
func midpoint(lo,hi int)int{return (lo+hi)/2}
func main(){lo,_:=strconv.Atoi(os.Args[1]);hi,_:=strconv.Atoi(os.Args[2]);fmt.Println(midpoint(lo,hi))}
```

### `idio:midpoint-overflow:rust` — signed_overflow (c->rust)

*Mirrors:* binary-search / merge midpoint `(lo+hi)/2` (the JDK / NIST-famous signed-overflow bug); idiomatic ports use wrapping (Rust) or 64-bit `int` (Go), so they are defined where C is UB.. *Witness:* `['2000000000', '2000000000']` (safe: `['10', '20']`).

```c
#include <stdio.h>
#include <stdlib.h>
static int midpoint(int lo,int hi){return (lo+hi)/2;}
int main(int argc,char**argv){int lo=atoi(argv[1]);int hi=atoi(argv[2]);printf("%d\n",midpoint(lo,hi));return 0;}
```

```rust
fn midpoint(lo:i32,hi:i32)->i32{ lo.wrapping_add(hi) / 2 }
fn main(){
  let lo: i32 = std::env::args().nth(1).unwrap().parse().unwrap();
  let hi: i32 = std::env::args().nth(2).unwrap().parse().unwrap();
  println!("{}", midpoint(lo,hi));
}
```

### `idio:rate-divide:go` — div_by_zero (c->go)

*Mirrors:* throughput/rate `total/count` (as in coreutils-style accounting); a zero divisor is UB in C, a defined panic in Rust and a defined panic in Go.. *Witness:* `['100', '0']` (safe: `['100', '4']`).

```c
#include <stdio.h>
#include <stdlib.h>
static int rate(int total,int count){return total/count;}
int main(int argc,char**argv){int t=atoi(argv[1]);int c=atoi(argv[2]);printf("%d\n",rate(t,c));return 0;}
```

```go
package main
import ("fmt";"os";"strconv")
func rate(total,count int)int{return total/count}
func main(){t,_:=strconv.Atoi(os.Args[1]);c,_:=strconv.Atoi(os.Args[2]);fmt.Println(rate(t,c))}
```

### `idio:rate-divide:rust` — div_by_zero (c->rust)

*Mirrors:* throughput/rate `total/count` (as in coreutils-style accounting); a zero divisor is UB in C, a defined panic in Rust and a defined panic in Go.. *Witness:* `['100', '0']` (safe: `['100', '4']`).

```c
#include <stdio.h>
#include <stdlib.h>
static int rate(int total,int count){return total/count;}
int main(int argc,char**argv){int t=atoi(argv[1]);int c=atoi(argv[2]);printf("%d\n",rate(t,c));return 0;}
```

```rust
fn rate(total:i32,count:i32)->i32{ total / count }
fn main(){
  let t: i32 = std::env::args().nth(1).unwrap().parse().unwrap();
  let c: i32 = std::env::args().nth(2).unwrap().parse().unwrap();
  println!("{}", rate(t,c));
}
```

### `multi:bitfield:go` — oversized_shift (c->go)

*Mirrors:* packed-struct bit-field extraction v<<w (out-of-range shift). *Witness:* `['1', '40']` (safe: `['1', '3']`).

```c
#include <stdio.h>
#include <stdlib.h>
static int field(int v,int w){return v<<w;}
int main(int argc,char**argv){int v=atoi(argv[1]);int w=atoi(argv[2]);printf("%d\n",field(v,w));return 0;}
```

```go
package main
import ("fmt";"os";"strconv")
func field(v,w int)int{return v<<uint(w)}
func main(){v,_:=strconv.Atoi(os.Args[1]);w,_:=strconv.Atoi(os.Args[2]);fmt.Println(field(v,w))}
```

### `multi:bitfield:rust` — oversized_shift (c->rust)

*Mirrors:* packed-struct bit-field extraction v<<w (out-of-range shift). *Witness:* `['1', '40']` (safe: `['1', '3']`).

```c
#include <stdio.h>
#include <stdlib.h>
static int field(int v,int w){return v<<w;}
int main(int argc,char**argv){int v=atoi(argv[1]);int w=atoi(argv[2]);printf("%d\n",field(v,w));return 0;}
```

```rust
fn field(v:i32,w:u32)->i32{ v.wrapping_shl(w) }
fn main(){
  let v: i32 = std::env::args().nth(1).unwrap().parse().unwrap();
  let w: u32 = std::env::args().nth(2).unwrap().parse().unwrap();
  println!("{}", field(v,w));
}
```

### `multi:bitfield:swift` — oversized_shift (c->swift)

*Mirrors:* packed-struct bit-field extraction v<<w (out-of-range shift). *Witness:* `['1', '40']` (safe: `['1', '3']`).

```c
#include <stdio.h>
#include <stdlib.h>
static int field(int v,int w){return v<<w;}
int main(int argc,char**argv){int v=atoi(argv[1]);int w=atoi(argv[2]);printf("%d\n",field(v,w));return 0;}
```

```swift
import Foundation
func field(_ v:Int32,_ w:Int32)->Int32{return v << w}
let v=Int32(CommandLine.arguments[1])!
let w=Int32(CommandLine.arguments[2])!
print(field(v,w))
```

### `multi:midpoint:go` — signed_overflow (c->go)

*Mirrors:* binary-search/merge midpoint (lo+hi)/2 signed-overflow bug. *Witness:* `['2000000000', '2000000000']` (safe: `['10', '20']`).

```c
#include <stdio.h>
#include <stdlib.h>
static int midpoint(int lo,int hi){return (lo+hi)/2;}
int main(int argc,char**argv){int lo=atoi(argv[1]);int hi=atoi(argv[2]);printf("%d\n",midpoint(lo,hi));return 0;}
```

```go
package main
import ("fmt";"os";"strconv")
func midpoint(lo,hi int)int{return (lo+hi)/2}
func main(){lo,_:=strconv.Atoi(os.Args[1]);hi,_:=strconv.Atoi(os.Args[2]);fmt.Println(midpoint(lo,hi))}
```

### `multi:midpoint:rust` — signed_overflow (c->rust)

*Mirrors:* binary-search/merge midpoint (lo+hi)/2 signed-overflow bug. *Witness:* `['2000000000', '2000000000']` (safe: `['10', '20']`).

```c
#include <stdio.h>
#include <stdlib.h>
static int midpoint(int lo,int hi){return (lo+hi)/2;}
int main(int argc,char**argv){int lo=atoi(argv[1]);int hi=atoi(argv[2]);printf("%d\n",midpoint(lo,hi));return 0;}
```

```rust
fn midpoint(lo:i32,hi:i32)->i32{ lo.wrapping_add(hi) / 2 }
fn main(){
  let lo: i32 = std::env::args().nth(1).unwrap().parse().unwrap();
  let hi: i32 = std::env::args().nth(2).unwrap().parse().unwrap();
  println!("{}", midpoint(lo,hi));
}
```

### `multi:midpoint:swift` — signed_overflow (c->swift)

*Mirrors:* binary-search/merge midpoint (lo+hi)/2 signed-overflow bug. *Witness:* `['2000000000', '2000000000']` (safe: `['10', '20']`).

```c
#include <stdio.h>
#include <stdlib.h>
static int midpoint(int lo,int hi){return (lo+hi)/2;}
int main(int argc,char**argv){int lo=atoi(argv[1]);int hi=atoi(argv[2]);printf("%d\n",midpoint(lo,hi));return 0;}
```

```swift
import Foundation
func midpoint(_ lo:Int32,_ hi:Int32)->Int32{return (lo &+ hi) / 2}
let lo=Int32(CommandLine.arguments[1])!
let hi=Int32(CommandLine.arguments[2])!
print(midpoint(lo,hi))
```

### `multi:rate:go` — div_by_zero (c->go)

*Mirrors:* coreutils-style throughput total/count (zero divisor). *Witness:* `['100', '0']` (safe: `['100', '4']`).

```c
#include <stdio.h>
#include <stdlib.h>
static int rate(int t,int c){return t/c;}
int main(int argc,char**argv){int t=atoi(argv[1]);int c=atoi(argv[2]);printf("%d\n",rate(t,c));return 0;}
```

```go
package main
import ("fmt";"os";"strconv")
func rate(t,c int)int{return t/c}
func main(){t,_:=strconv.Atoi(os.Args[1]);c,_:=strconv.Atoi(os.Args[2]);fmt.Println(rate(t,c))}
```

### `multi:rate:rust` — div_by_zero (c->rust)

*Mirrors:* coreutils-style throughput total/count (zero divisor). *Witness:* `['100', '0']` (safe: `['100', '4']`).

```c
#include <stdio.h>
#include <stdlib.h>
static int rate(int t,int c){return t/c;}
int main(int argc,char**argv){int t=atoi(argv[1]);int c=atoi(argv[2]);printf("%d\n",rate(t,c));return 0;}
```

```rust
fn rate(t:i32,c:i32)->i32{ t / c }
fn main(){
  let t: i32 = std::env::args().nth(1).unwrap().parse().unwrap();
  let c: i32 = std::env::args().nth(2).unwrap().parse().unwrap();
  println!("{}", rate(t,c));
}
```

### `multi:rate:swift` — div_by_zero (c->swift)

*Mirrors:* coreutils-style throughput total/count (zero divisor). *Witness:* `['100', '0']` (safe: `['100', '4']`).

```c
#include <stdio.h>
#include <stdlib.h>
static int rate(int t,int c){return t/c;}
int main(int argc,char**argv){int t=atoi(argv[1]);int c=atoi(argv[2]);printf("%d\n",rate(t,c));return 0;}
```

```swift
import Foundation
func rate(_ t:Int32,_ c:Int32)->Int32{return t / c}
let t=Int32(CommandLine.arguments[1])!
let c=Int32(CommandLine.arguments[2])!
print(rate(t,c))
```
