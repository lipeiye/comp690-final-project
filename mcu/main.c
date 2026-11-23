/* Measurement harness: run every embedded trial through the network and print its logits.
 *
 * Output (semihosting): "scratch <bytes>", then per trial "trial <i> <logit bytes in hex>", then
 * "stack <bytes>", the deepest stack use seen during any inference (the stack is painted before
 * each one). The trials live in their own .trials section and the harness code in .harness, so
 * the size of the deployed program can be read from the other sections. */
#include <stdint.h>
#include <string.h>

#include "dims.h"
#include "net.h"
#include "semihost.h"
#include "trials.h"

#define PAINT 0x5A5A5A5Au

extern uint32_t __stack_limit, __stack_top;

__attribute__((noinline, used)) void mark_start(void)
{
    __asm volatile("nop");
}

__attribute__((noinline, used)) void mark_stop(void)
{
    __asm volatile("nop\n\tnop");
}

HARNESS static void paint(void)
{
    uint32_t sp;
    __asm volatile("mov %0, sp" : "=r"(sp));
    for (uint32_t *p = &__stack_limit; (uintptr_t)p < sp - 64u; p++) {
        *p = PAINT;
    }
}

HARNESS static uint32_t stack_used(void)
{
    const uint32_t *p = &__stack_limit;
    while (p < &__stack_top && *p == PAINT) {
        p++;
    }
    return (uint32_t)((uintptr_t)&__stack_top - (uintptr_t)p);
}

HARNESS int main(void)
{
    int failed = net_init();
    put(failed ? "scratch too small " : "scratch ");
    put_dec(net_scratch_needed());
    put("\n");
    if (failed) {
        return 1;
    }
    uint32_t peak = 0;
    for (int i = 0; i < N_TRIALS; i++) {
        memcpy(net_input(), trials[i], sizeof trials[i]);
        paint();
        net_run();
        uint32_t used = stack_used();
        peak = used > peak ? used : peak;
        put("trial ");
        put_dec(i);
        put(" ");
        const uint8_t *out = net_output();
        for (int b = 0; b < N_CLASSES * OUT_BYTES; b++) {
            put_hex2(out[b]);
        }
        put("\n");
    }
    put("stack ");
    put_dec((int32_t)peak);
    put("\n");
    return 0;
}
