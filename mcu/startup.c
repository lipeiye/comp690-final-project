/* Cortex-M start-up for the QEMU boards: vector table, .data copied from flash, .bss cleared,
 * FPU and Helium (CP10/CP11) enabled where present, then main(). */
#include <stdint.h>

#include "semihost.h"

extern int main(void);
extern uint32_t __data_load__, __data_start__, __data_end__, __bss_start__, __bss_end__, __stack_top;

void Reset_Handler(void);

HARNESS static void Fault_Handler(void)
{
    put("fault\n");
    quit(1);
}

__attribute__((section(".vectors"), used)) void (*const vectors[16])(void) = {
    (void (*)(void))&__stack_top, Reset_Handler, Fault_Handler, Fault_Handler,
    Fault_Handler, Fault_Handler, Fault_Handler, 0, 0, 0, 0, Fault_Handler,
    Fault_Handler, 0, Fault_Handler, Fault_Handler,
};

void Reset_Handler(void)
{
    const uint32_t *src = &__data_load__;
    for (uint32_t *dst = &__data_start__; dst < &__data_end__;) {
        *dst++ = *src++;
    }
    for (uint32_t *dst = &__bss_start__; dst < &__bss_end__;) {
        *dst++ = 0;
    }
#if defined(__ARM_FP) || defined(__ARM_FEATURE_MVE)
    *(volatile uint32_t *)0xE000ED88 |= 0xFu << 20; /* CPACR: full access to CP10/CP11 (FPU, MVE) */
    __asm volatile("dsb\n\tisb");
#endif
    quit(main());
}
