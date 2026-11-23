/* Interface between the harness (main.c) and one network build (eegnet_int8.c or eegnet_f32.c).
 *
 * The network keeps two ping-pong activation buffers; the input window goes into the first one,
 * and each layer runs between mark_start() and mark_stop(), which the QEMU instruction counter
 * (icount.c) uses to count the instructions of every layer separately. */
#pragma once
#include <stdint.h>

void mark_start(void);
void mark_stop(void);

int net_init(void);              /* 0 on success; nonzero if the scratch buffer is too small */
int32_t net_scratch_needed(void); /* bytes of CMSIS-NN scratch the largest layer asks for */
void *net_input(void);           /* where the next input window goes */
const void *net_output(void);    /* N_CLASSES logits after net_run() */
void net_run(void);
