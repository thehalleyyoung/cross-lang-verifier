#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int overflow_add_one(int x) {
    return x + 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        return 2;
    }
    int x = (int)strtol(argv[1], NULL, 10);
    printf("%d\n", overflow_add_one(x));
    return 0;
}
