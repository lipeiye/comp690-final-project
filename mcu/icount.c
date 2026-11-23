/* QEMU TCG plugin (mcu_study): count guest instructions between two marker functions.
 * Arguments: start=<address of mark_start>, stop=<address of mark_stop>.
 * Prints one line per start/stop region and a total at exit (needs -d plugin). */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

static struct qemu_plugin_scoreboard *board;
static qemu_plugin_u64 count;
static uint64_t start_pc, stop_pc, began;
static unsigned regions;

static void at_start(unsigned int vcpu, void *udata)
{
    began = qemu_plugin_u64_get(count, vcpu);
}

static void at_stop(unsigned int vcpu, void *udata)
{
    char line[96];
    snprintf(line, sizeof line, "region %u: %" PRIu64 " instructions\n", regions++,
             qemu_plugin_u64_get(count, vcpu) - began);
    qemu_plugin_outs(line);
}

static void on_tb(struct qemu_plugin_tb *tb, void *udata)
{
    size_t n = qemu_plugin_tb_n_insns(tb);
    for (size_t i = 0; i < n; i++) {
        struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
        uint64_t pc = qemu_plugin_insn_vaddr(insn);
        if (pc == start_pc) {
            qemu_plugin_register_vcpu_insn_exec_cb(insn, at_start, QEMU_PLUGIN_CB_NO_REGS, NULL);
        }
        if (pc == stop_pc) {
            qemu_plugin_register_vcpu_insn_exec_cb(insn, at_stop, QEMU_PLUGIN_CB_NO_REGS, NULL);
        }
        qemu_plugin_register_vcpu_insn_exec_inline_per_vcpu(insn, QEMU_PLUGIN_INLINE_ADD_U64, count, 1);
    }
}

static void on_exit(void *udata)
{
    char line[64];
    snprintf(line, sizeof line, "total: %" PRIu64 " instructions\n", qemu_plugin_u64_sum(count));
    qemu_plugin_outs(line);
    qemu_plugin_scoreboard_free(board);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info, int argc,
                                           char **argv)
{
    for (int i = 0; i < argc; i++) {
        if (!strncmp(argv[i], "start=", 6)) {
            start_pc = strtoull(argv[i] + 6, NULL, 0) & ~1ull; /* drop the Thumb bit */
        } else if (!strncmp(argv[i], "stop=", 5)) {
            stop_pc = strtoull(argv[i] + 5, NULL, 0) & ~1ull;
        } else {
            fprintf(stderr, "icount: unknown argument %s\n", argv[i]);
            return -1;
        }
    }
    board = qemu_plugin_scoreboard_new(sizeof(uint64_t));
    count = qemu_plugin_scoreboard_u64(board);
    qemu_plugin_register_vcpu_tb_trans_cb(id, on_tb, NULL);
    qemu_plugin_register_atexit_cb(id, on_exit, NULL);
    return 0;
}
