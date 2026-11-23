/* Arm semihosting calls used by the harness: SYS_WRITE0 (0x04) and SYS_EXIT (0x18). */
#include "semihost.h"

static inline int32_t call(int32_t op, const void *arg)
{
    register int32_t r0 __asm__("r0") = op;
    register const void *r1 __asm__("r1") = arg;
    __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}

HARNESS void put(const char *text)
{
    call(0x04, text);
}

HARNESS void put_dec(int32_t value)
{
    char text[12];
    int i = 11;
    uint32_t magnitude = value < 0 ? (uint32_t)(-(int64_t)value) : (uint32_t)value;
    text[i] = 0;
    do {
        text[--i] = (char)('0' + magnitude % 10u);
        magnitude /= 10u;
    } while (magnitude);
    if (value < 0) {
        text[--i] = '-';
    }
    put(text + i);
}

HARNESS void put_hex2(uint8_t value)
{
    static const char digits[] = "0123456789abcdef";
    char text[3] = {digits[value >> 4], digits[value & 15u], 0};
    put(text);
}

HARNESS void quit(int failed)
{
    /* ADP_Stopped_ApplicationExit ends QEMU with status 0; RunTimeErrorUnknown with status 1 */
    call(0x18, (const void *)(failed ? 0x20024u : 0x20026u));
    for (;;) {
    }
}
