/*
 * Rust translation of union type punning patterns.
 * In Rust, union field access requires unsafe.
 * The idiomatic approach is transmute or to_bits()/from_bits().
 */

/// Float to bits: Rust has f32::to_bits() — safe, no union needed.
pub fn float_to_bits(value: f32) -> u32 {
    value.to_bits()  // safe, returns IEEE-754 representation
}

/// Bits to float: Rust has f32::from_bits() — safe.
pub fn bits_to_float(bits: u32) -> f32 {
    f32::from_bits(bits)  // safe, defined behavior
}

/// Network byte order: Rust uses u32::from_be_bytes / to_be_bytes.
/// No union type punning needed.
pub fn ntohl_manual(net_order: u32) -> u32 {
    let bytes = net_order.to_ne_bytes();
    ((bytes[0] as u32) << 24)
        | ((bytes[1] as u32) << 16)
        | ((bytes[2] as u32) << 8)
        | (bytes[3] as u32)
}

/// Tagged union: Rust enum is safe — discriminant is checked by compiler.
pub enum Variant {
    Int(i32),
    Float(f32),
}

pub fn extract_int(v: &Variant) -> i32 {
    match v {
        Variant::Int(i) => *i,
        Variant::Float(f) => *f as i32,  // explicit conversion, not type punning
    }
}
