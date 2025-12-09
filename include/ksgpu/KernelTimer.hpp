#ifndef _KSGPU_KERNEL_TIMER_HPP
#define _KSGPU_KERNEL_TIMER_HPP

#include "cuda_utils.hpp"  // CudaStreamWrapper, CUDA_CALL()
#include "time_utils.hpp"  // get_time(), time_diff()

#include <chrono>
#include <string>
#include <vector>


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
//   KernelTimer kt(niterations, nstreams);
//
//   while (kt.next()) {}
//       // NOTE: kernels should be launched on "kt.stream".
//       // NOTE: corresponding stream index is "kt.istream" (e.g. for pointer offsets)
//
//       mykernel <<< B, W, 0, kt.stream >>> (base_ptr + kt.istream * offset);
//       CUDA_PEEK("mykernel");
//
//       if (kt.warmed_up)
//           cout << "average time/kernel = " << kt.dt << ", sec" << endl;
//   }


struct KernelTimer {
    // State that the caller will want to reference (see above).
    double dt = 0.0;
    long istream = 0;
    bool warmed_up = false;
    cudaStream_t stream = nullptr;

    // "Internals"

    int nstreams = 0;
    int niterations = 0;
    long curr_iteration = -1;

    std::vector<CudaStreamWrapper> streams;  // length (nstreams)
    std::vector<CudaEventWrapper> events;    // length (nstreams)
    std::vector<double> timestamps;          // length (niterations+1)

    using Clock = std::chrono::steady_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    TimePoint start_time;

    KernelTimer(long niterations_, long nstreams_ = 1) :
        niterations(niterations_), nstreams(nstreams_)
    {
        if (nstreams <= 0)
            throw std::runtime_error("KernelTimer constructor called with nstreams <= 0");
        if (niterations < nstreams)
            throw std::runtime_error("KernelTimer constructor called with niterations < nstreams");

        streams = CudaStreamWrapper::create_vector(nstreams);
        events = CudaEventWrapper::create_vector(nstreams, ev_spin);
        timestamps.reserve(niterations+1);
    }

    bool next() 
    {
        if (curr_iteration < 0) {
            // Drain any pending kernels, before starting timing run.
            CUDA_CALL(cudaDeviceSynchronize());
            start_time = Clock::now();
        }
        else if (curr_iteration >= niterations)
            throw std::runtime_error("KernelTimer::next() called after timer has completed");

        curr_iteration++;
        istream = curr_iteration % nstreams;
        stream = streams[istream];

        if (curr_iteration >= nstreams) {
            // Busy-wait on event 'istream'.
            for (;;) {
                int status = cudaEventQuery(events[istream]);
                if (status == cudaSuccess)
                    break;
                if (status != cudaErrorNotReady)
                    throw std::runtime_error("cudaEventQuery() failed");
            }
        }

        timestamps.at(curr_iteration) = std::chrono::duration<double>(Clock::now() - start_time).count();

        int it_min = nstreams;

        if (curr_iteration > it_min) {
            int i0 = it_min + (curr_iteration-it_min)/2;
            dt = (timestamps.at(curr_iteration) - timestamps.at(i0)) / (curr_iteration-i0);
            warmed_up = true;
        }

        CUDA_CALL(cudaEventRecord(events[istream], stream));
        return (curr_iteration < niterations);
    }
};

// -------------------------------------------------------------------------------------------------
//
// KernelTimerOld: a simple header-only class for timing cuda kernels.
//
// Usage:
//
//   KernelTimerOld kt(nstreams);   // nstreams=1 is the default
//
//   for (int i = 0; i < niter; i++) {
//       // Launch kernel asynchronously on current stream.
//       mykernel <<< B, W, 0, kt.stream >>> ();
//       CUDA_PEEK("mykernel");
//
//       // KernelTimerOld.dt is the elapsed time **per call to advance()**.
//       // KernelTimerOld.advance() returns false for the first few "warmup" iterations.
//       // Note that KernelTimerOld.advance() synchronizes the next (not current) stream.
//
//       if (kt.advance())
//           cout << "average time/kernel = " << kt.dt << ", sec" << endl;
//   }


class KernelTimerOld {
public:
    KernelTimerOld(int nstreams=1);

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


inline KernelTimerOld::KernelTimerOld(int nstreams_) :
    nstreams(nstreams_)
{
    if (nstreams < 1)
        throw std::runtime_error("KernelTimerOld constructor called with nstreams < 1");

    streams = CudaStreamWrapper::create_vector(nstreams_);
    stream = streams.at(0);
    tv.reserve(1000);
}


inline bool KernelTimerOld::advance()
{
    tv.push_back(get_time());   // tv[nstreams]
    int n = niter++;

    // Infer time from time_diff(tv[m], tv[n])
    int mmin = nstreams + 1;
    int m = (n > mmin) ? (mmin + (n-mmin)/2) : max(n-1,0);
    dt = time_diff(tv[m],tv[n]) / max(n-m,1);

    istream = (niter % nstreams);
    stream = streams.at(istream);
    CUDA_CALL(cudaStreamSynchronize(stream));

    return (n > mmin);
}


} // namespace ksgpu

#endif // _KSGPU_KERNEL_TIMER_HPP
