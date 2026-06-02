/* c2rust corpus extraction unit: OpenSSL family, constant-time mask shift. */
int openssl_ct_mask(int v, int bits) {
    return v << bits;
}
