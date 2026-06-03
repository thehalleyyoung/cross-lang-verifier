#include <stdint.h>

int32_t sudo_tty_rate(int32_t bytes, int32_t elapsed) {
    return bytes / elapsed;
}
