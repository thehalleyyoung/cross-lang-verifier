/*
 * Rust translation of NULL pointer patterns.
 * Rust uses Option<&T> / Option<Box<T>> — no implicit NULL.
 * Raw pointer deref requires unsafe block.
 */

pub struct Node {
    pub value: i32,
    pub next: Option<Box<Node>>,
}

/// In Rust, node is Option — we must pattern match.
/// No UB: the compiler enforces the check.
pub fn get_value_if_valid(node: Option<&Node>) -> i32 {
    match node {
        Some(n) => n.value,
        None => -1,
    }
}

/// Linked list sum — Option makes the traversal safe.
pub fn sum_list(head: Option<&Node>) -> i32 {
    let mut sum = 0i32;
    let mut cur = head;
    while let Some(node) = cur {
        sum = sum.wrapping_add(node.value);
        cur = node.next.as_deref();
    }
    sum
}

/// String length: Option<&str> eliminates NULL ambiguity.
pub fn string_length_or_default(s: Option<&str>) -> i32 {
    match s {
        Some(string) => string.len() as i32,
        None => 0,
    }
}

/// Double-reference pattern: Rust uses Option to encode absence.
pub fn deref_result(pp: &Option<Box<Node>>) -> i32 {
    match pp {
        Some(p) => p.value,
        None => panic!("null pointer dereference"),  // explicit, not UB
    }
}
