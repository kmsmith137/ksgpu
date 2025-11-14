#ifndef _KSGPU_MEM_UTILS_HPP
#define _KSGPU_MEM_UTILS_HPP

#include <string>
#include <memory>
#include <type_traits>

#include "Dtype.hpp"
#include "rand_utils.hpp"     // randomize()
#include "xassert.hpp"

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Core functions, defined later in this file ("Implementation" below).


// See below for a complete list of 'af_alloc' flags.
// af_alloc() version 1: dtype is inferred from template parameter T.
template<typename T>
inline std::shared_ptr<T> af_alloc(long nelts, int flags);


// See below for a complete list of 'af_alloc' flags.
// af_alloc() version 2: dtype is specified at runtime.
// If (T != void) and dtype is inconsistent with native (C++) type T, then exception is thrown.
template<typename T> 
inline std::shared_ptr<T> af_alloc(Dtype dtype, long nelts, int flags);


template<typename T>
inline void af_copy(T *dst, int dst_flags, const T *src, int src_flags, long nelts);


template<typename T>
inline std::shared_ptr<T> af_clone(int dst_flags, const T *src, long nelts);


// -------------------------------------------------------------------------------------------------
//
// Flags for use in af_alloc().
//
// FIXME in hindsight, I should have "refined" the af_gpu flag, to indicate which GPU the
// memory "lives" on. Here is one scheme:
//
//   - Instead of indicating GPU ownership with:
//       aflags |= af_gpu;   // where af_gpu = 0x01
//
//     do this:
//       assert(0 <= current_cuda_device <= 253);
//       aflags |= (current_cuda_device+1);
//
//     Note that other flags need to be moved, e.g. af_uhost 0x2 -> 0x100, etc.
//
//   - The 'af_gpu' global is still defined, and equal to 255. This special value
//     means "current gpu", and is allowed in af_alloc() (maybe other places too?)
//
//   - The af_on_gpu() function (and Array<T:::on_gpu(), which is closely related)
//     becomes two functions, af_on_current_gpu(), and af_on_any_gpu().
//
//   - Direct flag bit-manipulations will need to be revisited, e.g.
//
//       // Test for memory being on a GPU.
//       // This still works if the caller intended "any GPU", but was "current GPU" intended?
//       if ((flags & af_gpu) != 0) ...;
//
//       // Never backwards-compatible (test always fails now).
//       if ((flags & af_gpu) == af_gpu) ... ;
//
//    - Can finally fix the issue where af_copy() just fails if the current device
//      is set incorrectly! (By setting the current device based on the flags, which
//      are already passed to af_copy().) Are there other places in ksgpu where a
//      similar issue exists, and can now be fixed?
//
// Note: could refine 'af_unified', to toggle cudaMemAttachHost vs cudaMemAttachGlobal.
// Note: could refine 'af_rhost' (e.g. cudaHostAllocWriteCombined).


// Location flags: where is memory allocated?
// Precisely one of these should be specified.
static constexpr int af_gpu = 0x01;      // gpu memory
static constexpr int af_uhost = 0x02;    // host memory, not registered with cuda runtime or page-locked
static constexpr int af_rhost = 0x04;    // host memory, registered with cuda runtime and page-locked
static constexpr int af_unified = 0x08;  // unified host+gpu memory (slow and never useful?)
static constexpr int af_location_flags = af_gpu | af_uhost | af_rhost | af_unified;

// Initialization flags
static constexpr int af_zero = 0x10;    // zero allocated memory
static constexpr int af_random = 0x20;  // randomize allocated memory
static constexpr int af_initialization_flags = af_zero | af_random;

// Mmap flags: if specified (with af_uhost or af_rhost), then mmap/munmap will be used instead of malloc/free.
static constexpr int af_mmap_small = 0x100;      // 4KB standard pages
static constexpr int af_mmap_huge = 0x200;       // 2MB huge pages
static constexpr int af_mmap_try_huge = 0x400;   // try huge pages, if fails then fall back to 4K pages, and print warning.
static constexpr int af_mmap_flags = af_mmap_small | af_mmap_huge | af_mmap_try_huge;

// Debugging flags
static constexpr int af_guard = 0x1000;     // creates "guard" region before/after allocated memory
static constexpr int af_verbose = 0x2000;   // prints messages on alloc/free
static constexpr int af_debug_flags = af_guard | af_verbose;
static constexpr int af_all_flags = af_location_flags | af_initialization_flags | af_mmap_flags | af_debug_flags;


// Throws exception if aflags are uninitialized or invalid.
extern void check_aflags(int aflags, const char *where = nullptr);

// Utility function for printing flags.
extern std::string aflag_str(int flags);

// Is memory addressable on GPU? On host?
inline bool af_on_gpu(int flags) { return (flags & (af_gpu | af_unified)) != 0; }
inline bool af_on_host(int flags) { return (flags & af_gpu) == 0; }


// -------------------------------------------------------------------------------------------------
//
// Implementation.


extern std::shared_ptr<void> _af_alloc(Dtype dtype, long nelts, int flags);


// Uses location flags, but ignores initialization and debug flags.
// Just a "thin" wrapper which selects between memcpy() or cudaMemcpy().
extern void _af_copy(void *dst, int dst_flags, const void *src, int src_flags, long nbytes);


// Version of af_alloc() without a runtime dtype.
template<typename T>
inline std::shared_ptr<T> af_alloc(long nelts, int flags)
{
    static_assert(!std::is_void_v<T>, "af_alloc<T>(): if T=void, then you must call the version of af_alloc() with a runtime dtype");
    return std::reinterpret_pointer_cast<T> (_af_alloc(Dtype::native<T>(), nelts, flags));
}


// Version of af_alloc() with a runtime dtype.
template<typename T>
inline std::shared_ptr<T> af_alloc(Dtype dtype, long nelts, int flags)
{
    _check_dtype<T> (dtype, "af_alloc");
    return std::reinterpret_pointer_cast<T> (_af_alloc(dtype, nelts, flags));
}


template<typename T>
inline void af_copy(T *dst, int dst_flags, const T *src, int src_flags, long nelts)
{
    // FIXME should have some static_asserts here, to ensure
    // that 'T' doesn't have constructors/destructors.

    xassert(nelts >= 0);
    long nbytes = nelts * sizeof(T);
    
    _af_copy(dst, dst_flags, src, src_flags, nbytes);
}


template<typename T>
inline std::shared_ptr<T> af_clone(int dst_flags, const T *src, long nelts)
{
    dst_flags &= ~af_initialization_flags;
    std::shared_ptr<T> ret = af_alloc<T> (nelts, dst_flags);
    af_copy(ret.get(), src, nelts);
}


} // namespace ksgpu

#endif  // _KSGPU_MEM_UTILS_HPP
