#ifndef _KSGPU_KERNEL_TIMER_HPP
#define _KSGPU_KERNEL_TIMER_HPP

#include <string>
#include <vector>

#include "cuda_utils.hpp"  // CudaStreamWrapper, CUDA_CALL()
#include "time_utils.hpp"


namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// KernelTimer: a simple header-only class for timing cuda kernels.
//
// Usage:
//
//   KernelTimer kt(nstreams);   // nstreams=1 is the default
//
//   for (int i = 0; i < niter; i++) {
//       // Launch kernel asynchronously on current stream.
//       mykernel <<< B, W, 0, kt.stream >>> ();
//       CUDA_PEEK("mykernel");
//
//       // KernelTimer.dt is the elapsed time **per call to advance()**.
//       // KernelTimer.advance() returns false for the first few "warmup" iterations.
//       // Note that KernelTimer.advance() synchronizes the next (not current) stream.
//
//       if (kt.advance())
//           cout << "average time/kernel = " << kt.dt << ", sec" << endl;
//   }


class KernelTimer {
public:
    KernelTimer(int nstreams=1);

    int niter = 0;                  // number of calls to advance() so far
    int istream = 0;                // always equal to (niter % nstreams)
    int nstreams = 0;               // specified at construction
    double dt = 0.0;                // elapsed time per call to advance()
    cudaStream_t stream = nullptr;  // always equal to streams[istream]

    bool advance();

    std::vector<CudaStreamWrapper> streams;  // length nstreams
    std::vector<struct timeval> tv;          // history of times when advance() was called
};


// -------------------------------------------------------------------------------------------------
//
// Implementation


KernelTimer::KernelTimer(int nstreams_) : nstreams(nstreams_)
{
    if (nstreams < 1)
        throw std::runtime_error("KernelTimer constructor called with nstreams < 1");

    streams.resize(nstreams);
    stream = streams.at(0);
    tv.reserve(1000);
}


bool KernelTimer::advance()
{
    tv.push_back(get_time());   // tv[nstreams]
    int n = niter++;

    // Infer time from time_diff(tv[m], tv[n])
    int mmin = nstreams;
    int m = (n > mmin) ? (mmin + (n-mmin)/2) : max(n-1,0);
    dt = time_diff(tv[m],tv[n]) / max(n-m,1);

    istream = (niter % nstreams);
    stream = streams.at(istream);
    CUDA_CALL(cudaStreamSynchronize(stream));

    return (n > mmin);
}


} // namespace ksgpu

#endif // _KSGPU_KERNEL_TIMER_HPP
