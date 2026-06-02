/* c2rust corpus extraction unit: Redis SDS family, header-room subtraction. */
int redis_sds_room(int x) {
    return x - 8;
}
