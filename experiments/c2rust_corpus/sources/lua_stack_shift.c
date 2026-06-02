/* c2rust corpus extraction unit: Lua family, stack-mask shift. */
int lua_stack_shift(int mask, int shift) {
    return mask << shift;
}
