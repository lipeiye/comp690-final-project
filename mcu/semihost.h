/* Minimal Arm semihosting (bkpt 0xAB): text output and exit, without the C library's stdio.
 * Everything here belongs to the measurement harness and is linked into its own .harness section,
 * so that it does not count towards the deployed program's flash. */
#pragma once
#include <stdint.h>

#define HARNESS __attribute__((section(".harness"), noinline))

HARNESS void put(const char *text);
HARNESS void put_dec(int32_t value);
HARNESS void put_hex2(uint8_t value);
HARNESS __attribute__((noreturn)) void quit(int failed);
