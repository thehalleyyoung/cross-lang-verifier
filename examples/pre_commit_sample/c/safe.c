#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int safe_identity(int x) {
    return x + 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        return 2;
    }
    int x = (int)strtol(argv[1], NULL, 10);
    printf("%d\n", safe_identity(x));
    return 0;
}
