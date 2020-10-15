
/*
 * Copyright (c) 2019-2020 Idein Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


#if defined(__arm__) && defined(__aarch64__)
#error "__arm__ and __aarch64__ are both defined"
#elif !defined(__arm__) && !defined(__aarch64__)
#error "__arm__ and __aarch64__ are both not defined"
#endif


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
