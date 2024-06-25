#include <stdio.h>
#include <algorithm>
#include <pthread.h>
#include <math.h>
#include <thread>

#include "CycleTimer.h"
#include "sqrt_ispc.h"
#include "immintrin.h"

using namespace ispc;
constexpr int SIMD_width = 8;

struct Arg {
    unsigned N;
    float initialGuess;
    float *values;
    float *output;
    unsigned threadId;
};

extern void sqrtSerial(int N, float startGuess, float* values, float* output);

static void verifyResult(int N, float* result, float* gold) {
    for (int i=0; i<N; i++) {
        if (fabs(result[i] - gold[i]) > 1e-4) {
            printf("Error: [%d] Got %f expected %f\n", i, result[i], gold[i]);
        }
    }
}

inline void print_m256(__m256 vec) {
    float* f = (float*)&vec;
    for(int i = 0; i < SIMD_width; ++i) {
        printf("%f ", f[i]);
    }
    putchar('\n');
}


const __m256 kThreshold = _mm256_set1_ps(0.00001f);
void sqrt_vx2(int N, float initialGuess, float values[], float output[]) {
    __m256 x;
    __m256 guess = _mm256_set1_ps(initialGuess);
    __m256 fabs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 one_ps = _mm256_set1_ps(1.f);
    __m256 three_ps = _mm256_set1_ps(3.f);
    __m256 point_five_ps = _mm256_set1_ps(0.5f);

    for(int i = 0; i < N; i += SIMD_width) {
        x = _mm256_load_ps(values + i);
        //value & 0x7fffffff = fabs(value)
        __m256 pred = _mm256_and_ps((guess * guess * x - one_ps), fabs_mask);
        
        __m256 mask = _mm256_cmp_ps(pred, kThreshold, _CMP_GT_OS);
        
        while(_mm256_movemask_ps(mask))  {
            __m256 new_guess = guess;
            new_guess = (three_ps * new_guess - x * new_guess * new_guess * new_guess) * point_five_ps;
            //update the necessary guess val
            guess = _mm256_blendv_ps(guess, new_guess, mask);
            
            pred = _mm256_and_ps((guess * guess * x - one_ps), fabs_mask);
            mask = _mm256_cmp_ps(pred, kThreshold, _CMP_GT_OS);
        }

        x = _mm256_load_ps(values + i);
        guess = _mm256_and_ps(guess, fabs_mask);
        _mm256_store_ps(output + i, guess * x);
    }
}

void workerThreadStart(Arg* const arg) {
    auto& [N, initialGuess, values, output, threadId] = *arg;
    sqrt_vx2(N, initialGuess, values + threadId * SIMD_width, output + threadId * SIMD_width);
}

int main() {

    const unsigned int N = 20 * 1000 * 1000;
    const float initialGuess = 1.0f;

    float* values = new alignas(32) float[N];
    float* output = new alignas(32) float[N];
    float* gold = new alignas(32) float[N];

    for (unsigned int i=0; i<N; i++)
    {
        // TODO: CS149 students.  Attempt to change the values in the
        // array here to meet the instructions in the handout: we want
        // to you generate best and worse-case speedups
        
        // starter code populates array with random input values
        values[i] = .001f + 2.998f * static_cast<float>(rand()) / RAND_MAX;
        // values[i] = 2.99f;
    }

    // generate a gold version to check results
    for (unsigned int i=0; i<N; i++)
        gold[i] = sqrt(values[i]);

    //
    // And run the serial implementation 3 times, again reporting the
    // minimum time.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrtSerial(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[sqrt serial]:\t\t[%.3f] ms\n", minSerial * 1000);

    verifyResult(N, output, gold);

    //
    // Compute the image using the ispc implementation; report the minimum
    // time of three runs.
    //
    double minISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minISPC = std::min(minISPC, endTime - startTime);
    }

    printf("[sqrt ispc]:\t\t[%.3f] ms\n", minISPC * 1000);

    verifyResult(N, output, gold);

    // Clear out the buffer
    for (unsigned int i = 0; i < N; ++i)
        output[i] = 0;

    //
    // Tasking version of the ISPC code
    //
    double minTaskISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc_withtasks(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minTaskISPC = std::min(minTaskISPC, endTime - startTime);
    }

    printf("[sqrt task ispc]:\t[%.3f] ms\n", minTaskISPC * 1000);

    verifyResult(N, output, gold);

    std::fill(output, output + N, 0);
    double minIntrinsics = 1e30;
    double startTime = CycleTimer::currentSeconds();
    sqrt_vx2(N, initialGuess, values, output);
    double endTime = CycleTimer::currentSeconds();
    minIntrinsics = std::min(minIntrinsics, endTime - startTime);
    
    printf("[intrinsics]:\t\t[%.3f] ms\n", minIntrinsics * 1000);
    verifyResult(N, output, gold);

    double minThreadIntrinsics = 1e30;
    startTime = CycleTimer::currentSeconds();
    
    unsigned numThreads = 8;
    Arg args[numThreads];
    std::thread workers[numThreads];
    unsigned partsNum = N / numThreads;
    for(unsigned i = 0; i < numThreads; ++i) {
        args[i] = {partsNum, initialGuess, values, output, i};
    }
    for(unsigned i = 1; i < numThreads; ++i) {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }
    workerThreadStart(&args[0]);
    for(unsigned i = 1; i < numThreads; ++i) {
        workers[i].join();
    }
    
    endTime = CycleTimer::currentSeconds();
    minThreadIntrinsics = std::min(minThreadIntrinsics, endTime - startTime);
    printf("[multiple threads intrinsics]:\t\t[%.3f] ms\n", minThreadIntrinsics * 1000);
    verifyResult(N, output, gold);
    

    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minSerial/minISPC);
    printf("\t\t\t\t(%.2fx speedup from task ISPC)\n", minSerial/minTaskISPC);
    printf("\t\t\t\t(%.2fx speedup from Intrinsics)\n", minSerial/minIntrinsics);
    printf("\t\t\t\t(%.2fx speedup from multiple threads Intrinsics)\n", minSerial/minThreadIntrinsics);   
    delete [] values;
    delete [] output;
    delete [] gold;

    return 0;
}
