/* c2rust corpus extraction unit: XZ Utils family, bounded range shift. */
int xz_range_shift(int symbol, int bits) {
    return symbol << bits;
}
