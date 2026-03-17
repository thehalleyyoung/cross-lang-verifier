/*
 * NULL pointer divergence — inspired by CVE-2009-1897 (Linux kernel NULL deref)
 * and CVE-2018-1000001 (glibc realpath NULL deref).
 *
 * In C, dereferencing NULL is UB. Compilers can optimize away NULL checks
 * that follow a dereference (since UB means it "can't" be NULL).
 * In Rust, Option<&T> makes nullability explicit; raw pointer deref requires unsafe.
 */
#include <stddef.h>
#include <string.h>

struct Node {
    int value;
    struct Node *next;
};

/* CVE-style: dereference before NULL check (compiler may elide the check). */
int get_value_if_valid(struct Node *node) {
    int val = node->value;   /* dereference first — UB if node is NULL */
    if (node == NULL)         /* compiler may optimize this away */
        return -1;
    return val;
}

/* Linked list traversal: no NULL guard on inner access. */
int sum_list(struct Node *head) {
    int sum = 0;
    struct Node *cur = head;
    while (cur != NULL) {
        sum += cur->value;
        cur = cur->next;
    }
    return sum;
}

/* String length with potential NULL input (real pattern from glibc bugs). */
int string_length_or_default(const char *s) {
    if (s == NULL) return 0;
    return (int)strlen(s);
}

/* Double-pointer pattern: *pp may be NULL after lookup. */
int deref_result(struct Node **pp) {
    struct Node *p = *pp;
    return p->value;  /* UB if *pp was NULL */
}
