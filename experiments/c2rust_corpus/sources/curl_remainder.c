/* c2rust corpus extraction unit: curl family, chunk remainder. */
int curl_remainder(int bytes, int chunk) {
    return bytes % chunk;
}
