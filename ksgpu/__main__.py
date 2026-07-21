import re
import sys
import argparse

import argcomplete

from . import ksgpu_pybind11
from . import tests
from . import timing


#########################################   test command  ##########################################


def parse_test(subparsers):
    help_text = "Run unit tests (use flags to select specific tests)"
    parser = subparsers.add_parser("test", help=help_text, description=help_text)
    parser.add_argument('-g', '--gpu', type=int, default=0, help="GPU to use for tests (default 0)")
    parser.add_argument('--arr', action='store_true', help='Runs test_array() (Array strides/reshape/fill/convert)')
    parser.add_argument('--dtk', action='store_true', help='Runs test_device_transpose_kernels()')
    parser.add_argument('--mcpy', action='store_true', help='Runs test_memcpy_kernels()')
    parser.add_argument('--smma', action='store_true', help='Runs test_sparse_mma()')
    parser.add_argument('--conv', action='store_true', help='Runs numpy/cupy <-> C++ array-conversion tests')
    parser.add_argument('--dtype', action='store_true', help='Runs numpy.dtype <-> ksgpu::Dtype conversion tests')
    parser.add_argument('--stream', action='store_true', help='Runs CudaStreamWrapper / cupy stream interop tests')


def _run_counted(f):
    """Runs a python test-runner function which returns (num_passed, num_total)."""
    passed, total = f()
    if passed < total:
        raise RuntimeError(f'{f.__module__}.{f.__name__}: {total-passed}/{total} tests failed')


def test(args):
    test_flags = [ 'arr', 'dtk', 'mcpy', 'smma', 'conv', 'dtype', 'stream' ]
    run_all_tests = not any(getattr(args,x) for x in test_flags)

    ksgpu_pybind11.set_cuda_device(args.gpu)

    if run_all_tests or args.arr:
        tests.test_array()

    if run_all_tests or args.dtk:
        tests.test_device_transpose_kernels()

    if run_all_tests or args.mcpy:
        tests.test_memcpy_kernels()

    if run_all_tests or args.smma:
        tests.test_sparse_mma()

    # The remaining tests are python-side (local imports, since the modules import cupy).

    if run_all_tests or args.conv:
        from . import test_array_conversion
        _run_counted(test_array_conversion.run_numpy_tests)
        _run_counted(test_array_conversion.run_cupy_tests)

    if run_all_tests or args.dtype:
        from . import test_dtype
        _run_counted(test_dtype.run_roundtrip_tests)
        _run_counted(test_dtype.run_error_tests)
        _run_counted(test_dtype.run_argument_tests)
        _run_counted(test_dtype.run_cupy_tests)

    if run_all_tests or args.stream:
        from . import test_stream_conversion
        _run_counted(test_stream_conversion.run_all_tests)


#########################################   time command  ##########################################


def parse_time(subparsers):
    help_text = "Run timings (use flags to select specific timings)"
    parser = subparsers.add_parser("time", help=help_text, description=help_text)
    parser.add_argument('-g', '--gpu', type=int, default=0, help="GPU to use for timing (default 0)")
    parser.add_argument('--atom', action='store_true', help='Runs time_atomic_add()')
    parser.add_argument('--fma', action='store_true', help='Runs time_fma()')
    parser.add_argument('--gmem', action='store_true', help='Runs time_global_memory()')
    parser.add_argument('--l2', action='store_true', help='Runs time_l2_cache()')
    parser.add_argument('--ltra', action='store_true', help='Runs time_local_transpose()')
    parser.add_argument('--mcpy', action='store_true', help='Runs time_memcpy_kernels()')
    parser.add_argument('--shmem', action='store_true', help='Runs time_shared_memory()')
    parser.add_argument('--tc', action='store_true', help='Runs time_tensor_cores()')
    parser.add_argument('--wshuf', action='store_true', help='Runs time_warp_shuffle()')


def time_command(args):
    timing_flags = [ 'atom', 'fma', 'gmem', 'l2', 'ltra', 'mcpy', 'shmem', 'tc', 'wshuf' ]
    run_all_timings = not any(getattr(args,x) for x in timing_flags)

    ksgpu_pybind11.set_cuda_device(args.gpu)

    if run_all_timings or args.atom:
        timing.time_atomic_add()
    if run_all_timings or args.fma:
        timing.time_fma()
    if run_all_timings or args.gmem:
        timing.time_global_memory()
    if run_all_timings or args.l2:
        timing.time_l2_cache()
    if run_all_timings or args.ltra:
        timing.time_local_transpose()
    if run_all_timings or args.mcpy:
        timing.time_memcpy_kernels()
    if run_all_timings or args.shmem:
        timing.time_shared_memory()
    if run_all_timings or args.tc:
        timing.time_tensor_cores()
    if run_all_timings or args.wshuf:
        timing.time_warp_shuffle()


#####################################   show_devices command  ######################################


def parse_show_devices(subparsers):
    help_text = "Show cuda devices, with properties (compute capability, clocks, memory, bandwidth)"
    subparsers.add_parser("show_devices", help=help_text, description=help_text)

def show_devices(args):
    ksgpu_pybind11.show_devices()


##################################   reverse_engineer_mma command  #################################


def parse_reverse_engineer_mma(subparsers):
    help_text = "Print register <-> matrix-element mappings for mma.* PTX instructions"
    subparsers.add_parser("reverse_engineer_mma", help=help_text, description=help_text)

def reverse_engineer_mma(args):
    ksgpu_pybind11.reverse_engineer_mma()


#######################################   scratch command  #########################################


def parse_scratch(subparsers):
    help_text = "For debugging: run whatever code is currently in src_lib/scratch.cu"
    subparsers.add_parser("scratch", help=help_text, description=help_text)

def scratch(args):
    # The scratch() function is defined in src_lib/scratch.cu.
    ksgpu_pybind11.scratch()


####################################################################################################


class _KsgpuParser(argparse.ArgumentParser):
    """ArgumentParser variant that swallows argparse's auto-appended
    '(choose from {...})' in invalid-choice errors and points the user at
    --help instead. Pairs with metavar='command' on add_subparsers() so
    the run-on choices listing also disappears from --help / usage."""
    def error(self, message):
        # Strip the "(choose from ...)" suffix argparse appends on
        # invalid-subcommand errors. Wording is fragile across Python
        # versions; falls through harmlessly if argparse changes it.
        message = re.sub(r" \(choose from .*\)$", "", message)
        self.print_usage(sys.stderr)
        sys.stderr.write(f"{self.prog}: error: {message}\n")
        sys.stderr.write(f"For a list of all commands, see '{self.prog} --help'.\n")
        sys.exit(2)


def get_parser():
    """
    Create and return the argument parser for ksgpu.

    This function is separate from main() so that sphinx-argparse can
    introspect the parser without actually parsing command-line arguments.
    """
    parser = _KsgpuParser(description="ksgpu command-line driver (use --help for more info)")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")

    parse_test(subparsers)
    parse_time(subparsers)
    parse_show_devices(subparsers)
    parse_reverse_engineer_mma(subparsers)
    parse_scratch(subparsers)

    return parser


def main():
    ksgpu_pybind11.seed_default_rng(137)   # reproducible run; remove for full randomness

    parser = get_parser()
    argcomplete.autocomplete(parser)

    args = parser.parse_args()

    if args.command == "test":
        test(args)
    elif args.command == "time":
        time_command(args)
    elif args.command == "show_devices":
        show_devices(args)
    elif args.command == "reverse_engineer_mma":
        reverse_engineer_mma(args)
    elif args.command == "scratch":
        scratch(args)
    else:
        print(f"Command '{args.command}' not recognized", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
