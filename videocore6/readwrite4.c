#include <stdint.h>


uint32_t read4(void * const addr)
{
    uint32_t value;

    asm volatile (
#if defined(__arm__)
            "ldr %[value], [%[addr]]\n\t"
#elif defined(__aarch64__)
            "ldr %w[value], [%[addr]]\n\t"
#endif
            : [value] "=r" (value)
            : [addr] "r" (addr)
            : "memory"
    );

    return value;
}


void write4(void * const addr, const uint32_t value)
{
    asm volatile (
#if defined(__arm__)
            "str %[value], [%[addr]]\n\t"
#elif defined(__aarch64__)
            "str %w[value], [%[addr]]\n\t"
#endif
            :
            : [value] "r" (value),
              [addr] "r" (addr)
            : "memory"
    );
}
