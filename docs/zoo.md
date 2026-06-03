# The divergence zoo

*Auto-generated from the live corpora — do not edit by hand; run `python -m ub_oracle.divergence_zoo`.*

A machine-readable, indexed catalogue of the cross-language divergence patterns this tool catches, **indexed by class and language pair**. Every divergent exhibit carries a concrete **witnessing input** and is **re-confirmed live** by `confirm_zoo()` (the oracle must still flag the divergence on the witness and stay silent on the safe input).

*content hash: `29a67003bcb5a9c3` — 18 divergent exhibits across 5 classes.*

## Index — class × pair

| divergence class | language pair | exhibits |
|------------------|---------------|----------|
| `div_by_zero` | `c->go` | `idio:rate-divide:go`, `multi:rate:go` |
| `div_by_zero` | `c->rust` | `idio:rate-divide:rust`, `multi:rate:rust` |
| `div_by_zero` | `c->swift` | `multi:rate:swift` |
| `memcpy_overlap` | `c->go` | `idio:memcpy-overlap:go` |
| `memcpy_overlap` | `c->rust` | `idio:memcpy-overlap:rust` |
| `oversized_shift` | `c->go` | `multi:bitfield:go` |
| `oversized_shift` | `c->rust` | `idio:bitfield-shift:rust`, `multi:bitfield:rust` |
| `oversized_shift` | `c->swift` | `multi:bitfield:swift` |
| `signed_overflow` | `c->go` | `idio:midpoint-overflow:go`, `multi:midpoint:go` |
| `signed_overflow` | `c->rust` | `idio:midpoint-overflow:rust`, `multi:midpoint:rust` |
| `signed_overflow` | `c->swift` | `multi:midpoint:swift` |
| `uninit_padding` | `c->go` | `idio:uninit-padding:go` |
| `uninit_padding` | `c->rust` | `idio:uninit-padding:rust` |

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

### `idio:memcpy-overlap:go` — memcpy_overlap (c->go)

*Mirrors:* in-place buffer shift written with `memcpy` instead of `memmove`; the overlapping C call is UB, while Rust `copy_within` and Go `copy` have defined memmove-like slice semantics.. *Witness:* `['1', '0', '4']` (safe: `['8', '0', '4']`).

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifdef CLV_CHECK_MEMCPY
static void *clv_checked_memcpy(void *dst,const void *src,size_t n){uintptr_t d=(uintptr_t)dst;uintptr_t s=(uintptr_t)src;if(n>0&&d<s+n&&s<d+n){fprintf(stderr,"runtime error: memcpy-param-overlap dst=%p src=%p n=%zu\n",dst,src,n);abort();}return memmove(dst,src,n);}
#define memcpy(d,s,n) clv_checked_memcpy((d),(s),(n))
#endif
static void shift(char *buf,int dst,int src,int n){memcpy(buf+dst,buf+src,(size_t)n);}
int main(int argc,char**argv){int dst=atoi(argv[1]);int src=atoi(argv[2]);int n=atoi(argv[3]);char buf[17]="ABCDEFGHIJKLMNOP";if(dst<0||src<0||n<0||dst+n>16||src+n>16)return 3;shift(buf,dst,src,n);printf("%s\n",buf);return 0;}
```

```go
package main
import ("fmt";"os";"strconv")
func main(){dst,_:=strconv.Atoi(os.Args[1]);src,_:=strconv.Atoi(os.Args[2]);n,_:=strconv.Atoi(os.Args[3]);buf:=[]byte("ABCDEFGHIJKLMNOP");copy(buf[dst:dst+n],buf[src:src+n]);fmt.Println(string(buf))}
```

### `idio:memcpy-overlap:rust` — memcpy_overlap (c->rust)

*Mirrors:* in-place buffer shift written with `memcpy` instead of `memmove`; the overlapping C call is UB, while Rust `copy_within` and Go `copy` have defined memmove-like slice semantics.. *Witness:* `['1', '0', '4']` (safe: `['8', '0', '4']`).

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifdef CLV_CHECK_MEMCPY
static void *clv_checked_memcpy(void *dst,const void *src,size_t n){uintptr_t d=(uintptr_t)dst;uintptr_t s=(uintptr_t)src;if(n>0&&d<s+n&&s<d+n){fprintf(stderr,"runtime error: memcpy-param-overlap dst=%p src=%p n=%zu\n",dst,src,n);abort();}return memmove(dst,src,n);}
#define memcpy(d,s,n) clv_checked_memcpy((d),(s),(n))
#endif
static void shift(char *buf,int dst,int src,int n){memcpy(buf+dst,buf+src,(size_t)n);}
int main(int argc,char**argv){int dst=atoi(argv[1]);int src=atoi(argv[2]);int n=atoi(argv[3]);char buf[17]="ABCDEFGHIJKLMNOP";if(dst<0||src<0||n<0||dst+n>16||src+n>16)return 3;shift(buf,dst,src,n);printf("%s\n",buf);return 0;}
```

```rust
fn main(){
  let a: Vec<String> = std::env::args().collect();
  let dst: usize = a[1].parse().unwrap();
  let src: usize = a[2].parse().unwrap();
  let n: usize = a[3].parse().unwrap();
  let mut buf = b"ABCDEFGHIJKLMNOP".to_vec();
  buf.copy_within(src..src+n, dst);
  println!("{}", String::from_utf8_lossy(&buf));
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

### `idio:uninit-padding:go` — uninit_padding (c->go)

*Mirrors:* whole-struct byte serialization after assigning fields; C padding bytes are indeterminate, while safe Rust/Go serializers start from zeroed bytes and write only fields.. *Witness:* `['7', '16909060', '1']` (safe: `['7', '16909060', '0']`).

```c
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
struct P {
    uint8_t tag;
    uint32_t value;
};
_Static_assert(sizeof(struct P) == 8, "struct size drift");
_Static_assert(sizeof(struct P) > 5, "no padding in struct P");
_Static_assert(offsetof(struct P, tag) == 0, "tag offset drift");
_Static_assert(offsetof(struct P, value) == 4, "value offset drift");
__attribute__((noinline)) static uint32_t digest(unsigned long long v0, unsigned long long v1, int expose_padding){
    struct P p;
    if (!expose_padding) memset(&p, 0, sizeof p);
#ifdef CLV_ZERO_PADDING
    memset(&p, 0, sizeof p);
#endif
    p.tag = (uint8_t)v0;
    p.value = (uint32_t)v1;
    unsigned char bytes[sizeof p];
    memcpy(bytes, &p, sizeof p);
    uint32_t acc = 0;
    for (size_t i = 0; i < sizeof bytes; ++i) acc = acc * 131u + bytes[i];
    return acc;
}
int main(int argc, char **argv){
    unsigned long long v0 = argc > 1 ? strtoull(argv[1], 0, 10) : 7ull;
    unsigned long long v1 = argc > 2 ? strtoull(argv[2], 0, 10) : 16909060ull;
    int expose_padding = argc > 3 ? atoi(argv[3]) : 1;
    printf("%u\n", digest(v0, v1, expose_padding));
    return 0;
}
```

```go
package main
import (
	"encoding/binary"
	"fmt"
	"os"
	"strconv"
)
func main() {
	bytes := make([]byte, 8)
	v0 := uint64(7)
	if len(os.Args) > 1 { parsed, _ := strconv.ParseUint(os.Args[1], 10, 64); v0 = parsed }
	bytes[0] = byte(v0)
	v1 := uint64(16909060)
	if len(os.Args) > 2 { parsed, _ := strconv.ParseUint(os.Args[2], 10, 64); v1 = parsed }
	binary.LittleEndian.PutUint32(bytes[4:8], uint32(v1))
	var acc uint32
	for _, b := range bytes { acc = acc*131 + uint32(b) }
	fmt.Println(acc)
}
```

### `idio:uninit-padding:rust` — uninit_padding (c->rust)

*Mirrors:* whole-struct byte serialization after assigning fields; C padding bytes are indeterminate, while safe Rust/Go serializers start from zeroed bytes and write only fields.. *Witness:* `['7', '16909060', '1']` (safe: `['7', '16909060', '0']`).

```c
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
struct P {
    uint8_t tag;
    uint32_t value;
};
_Static_assert(sizeof(struct P) == 8, "struct size drift");
_Static_assert(sizeof(struct P) > 5, "no padding in struct P");
_Static_assert(offsetof(struct P, tag) == 0, "tag offset drift");
_Static_assert(offsetof(struct P, value) == 4, "value offset drift");
__attribute__((noinline)) static uint32_t digest(unsigned long long v0, unsigned long long v1, int expose_padding){
    struct P p;
    if (!expose_padding) memset(&p, 0, sizeof p);
#ifdef CLV_ZERO_PADDING
    memset(&p, 0, sizeof p);
#endif
    p.tag = (uint8_t)v0;
    p.value = (uint32_t)v1;
    unsigned char bytes[sizeof p];
    memcpy(bytes, &p, sizeof p);
    uint32_t acc = 0;
    for (size_t i = 0; i < sizeof bytes; ++i) acc = acc * 131u + bytes[i];
    return acc;
}
int main(int argc, char **argv){
    unsigned long long v0 = argc > 1 ? strtoull(argv[1], 0, 10) : 7ull;
    unsigned long long v1 = argc > 2 ? strtoull(argv[2], 0, 10) : 16909060ull;
    int expose_padding = argc > 3 ? atoi(argv[3]) : 1;
    printf("%u\n", digest(v0, v1, expose_padding));
    return 0;
}
```

```rust
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut bytes = [0u8; 8];
    let v0: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(7);
    bytes[0] = v0 as u8;
    let v1: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16909060);
    bytes[4..8].copy_from_slice(&((v1 as u32).to_le_bytes()));
    let mut acc: u32 = 0;
    for b in bytes { acc = acc.wrapping_mul(131).wrapping_add(b as u32); }
    println!("{}", acc);
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
