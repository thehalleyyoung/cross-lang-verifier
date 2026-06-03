#include <stdint.h>

int32_t zlib_checksum_window(int32_t sum) {
    return sum + 65521;
}
