#ifndef _KSGPU_COMMAND_LINE_INTERFACE_HPP
#define _KSGPU_COMMAND_LINE_INTERFACE_HPP

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// Entry points for the 'ksgpu' command-line driver ('python -m ksgpu <command>').
//
// Each function is implemented in the correspondingly-named src_lib/*.cu file,
// python-bound in src_pybind11/ksgpu_pybind11.cpp, and dispatched from
// ksgpu/__main__.py. None of these functions seed the default RNG -- the
// command-line driver seeds it once at startup (for a reproducible run).


// Unit tests ('python -m ksgpu test').
extern void test_array(long niter = 1000, bool noisy = false);
extern void test_device_transpose_kernels();
extern void test_memcpy_kernels();
extern void test_sparse_mma(long niter = 100);

// Timings ('python -m ksgpu time').
extern void time_atomic_add();
extern void time_fma();
extern void time_global_memory();
extern void time_l2_cache();
extern void time_local_transpose();
extern void time_memcpy_kernels();
extern void time_shared_memory();
extern void time_tensor_cores();
extern void time_warp_shuffle();

// Each of these gets its own command (e.g. 'python -m ksgpu show_devices').
extern void show_devices();
extern void reverse_engineer_mma();
extern void scratch();


}  // namespace ksgpu

#endif // _KSGPU_COMMAND_LINE_INTERFACE_HPP
