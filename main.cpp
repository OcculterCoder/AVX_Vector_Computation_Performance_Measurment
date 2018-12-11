#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1300)

#include <immintrin.h>

int check_4th_gen_intel_core_features()
{
    const int the_4th_gen_features =
        (_FEATURE_AVX2 | _FEATURE_FMA | _FEATURE_BMI | _FEATURE_LZCNT | _FEATURE_MOVBE);
    return _may_i_use_cpu_feature( the_4th_gen_features );
}

#else /* non-Intel compiler */

#include <stdint.h>
#if defined(_MSC_VER)
# include <intrin.h>
#endif

void run_cpuid(uint32_t eax, uint32_t ecx, uint32_t* abcd)
{
#if defined(_MSC_VER)
    __cpuidex(abcd, eax, ecx);
#else
    uint32_t ebx, edx;
# if defined( __i386__ ) && defined ( __PIC__ )
     /* in case of PIC under 32-bit EBX cannot be clobbered */
    __asm__ ( "movl %%ebx, %%edi \n\t cpuid \n\t xchgl %%ebx, %%edi" : "=D" (ebx),
# else
    __asm__ ( "cpuid" : "+b" (ebx),
# endif
              "+a" (eax), "+c" (ecx), "=d" (edx) );
    abcd[0] = eax; abcd[1] = ebx; abcd[2] = ecx; abcd[3] = edx;
#endif
}

int check_xcr0_ymm()
{
    uint32_t xcr0;
#if defined(_MSC_VER)
    xcr0 = (uint32_t)_xgetbv(0);  /* min VS2010 SP1 compiler is required */
#else
    __asm__ ("xgetbv" : "=a" (xcr0) : "c" (0) : "%edx" );
#endif
    return ((xcr0 & 6) == 6); /* checking if xmm and ymm state are enabled in XCR0 */
}


int check_4th_gen_intel_core_features()
{
    uint32_t abcd[4];
    uint32_t fma_movbe_osxsave_mask = ((1 << 12) | (1 << 22) | (1 << 27));
    uint32_t avx2_bmi12_mask = (1 << 5) | (1 << 3) | (1 << 8);

    /* CPUID.(EAX=01H, ECX=0H):ECX.FMA[bit 12]==1   &&
       CPUID.(EAX=01H, ECX=0H):ECX.MOVBE[bit 22]==1 &&
       CPUID.(EAX=01H, ECX=0H):ECX.OSXSAVE[bit 27]==1 */
    run_cpuid( 1, 0, abcd );
    if ( (abcd[2] & fma_movbe_osxsave_mask) != fma_movbe_osxsave_mask )
        return 0;

    if ( ! check_xcr0_ymm() )
        return 0;

    /*  CPUID.(EAX=07H, ECX=0H):EBX.AVX2[bit 5]==1  &&
        CPUID.(EAX=07H, ECX=0H):EBX.BMI1[bit 3]==1  &&
        CPUID.(EAX=07H, ECX=0H):EBX.BMI2[bit 8]==1  */
    run_cpuid( 7, 0, abcd );
    if ( (abcd[1] & avx2_bmi12_mask) != avx2_bmi12_mask )
        return 0;

    /* CPUID.(EAX=80000001H):ECX.LZCNT[bit 5]==1 */
    run_cpuid( 0x80000001, 0, abcd );
    if ( (abcd[2] & (1 << 5)) == 0)
        return 0;

    return 1;
}

#endif /* non-Intel compiler */

#include <iostream>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <sys/time.h>
using namespace std;

static int can_use_intel_core_4th_gen_features()
{
    static int the_4th_gen_features_available = -1;
    /* test is performed once */
    if (the_4th_gen_features_available < 0 )
        the_4th_gen_features_available = check_4th_gen_intel_core_features();

    return the_4th_gen_features_available;
}

inline double timestamp() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return double(tp.tv_sec) + tp.tv_usec / 1000000.;
}

inline void AVXsum64(float *a, float *b, float *c, int ARR_SIZE)
{
    for (int i=0; i < ARR_SIZE ; i+=64){

        __m256 res0 __attribute__(( aligned(32))) = _mm256_add_ps(_mm256_load_ps(&a[i]),_mm256_load_ps(&b[i])); // adding 8 values of array a and b

        _mm256_store_ps(&c[i],res0);

        __m256 res1 __attribute__(( aligned(32))) = _mm256_add_ps(_mm256_load_ps(&a[i+8]),_mm256_load_ps(&b[i+8])); // adding 8 values of array a and b

        _mm256_store_ps(&c[i+8],res1);

        __m256 res2 __attribute__(( aligned(32))) = _mm256_add_ps(_mm256_load_ps(&a[i+16]),_mm256_load_ps(&b[i+16])); // adding 8 values of array a and b

        _mm256_store_ps(&c[i+16],res2);

        __m256 res3 __attribute__(( aligned(32))) = _mm256_add_ps(_mm256_load_ps(&a[i+24]),_mm256_load_ps(&b[i+24])); // adding 8 values of array a and b

        _mm256_store_ps(&c[i+24],res3);

        __m256 res4 __attribute__(( aligned(32))) = _mm256_add_ps(_mm256_load_ps(&a[i+32]),_mm256_load_ps(&b[i+32])); // adding 8 values of array a and b

        _mm256_store_ps(&c[i+32],res4);

        __m256 res5 __attribute__(( aligned(32))) = _mm256_add_ps(_mm256_load_ps(&a[i+40]),_mm256_load_ps(&b[i+40])); // adding 8 values of array a and b

        _mm256_store_ps(&c[i+40],res5);

        __m256 res6 __attribute__(( aligned(32))) = _mm256_add_ps(_mm256_load_ps(&a[i+48]),_mm256_load_ps(&b[i+48])); // adding 8 values of array a and b

        _mm256_store_ps(&c[i+48],res6);

        __m256 res7 __attribute__(( aligned(32))) = _mm256_add_ps(_mm256_load_ps(&a[i+56]),_mm256_load_ps(&b[i+56])); // adding 8 values of array a and b

        _mm256_store_ps(&c[i+56],res7);

    }
}

inline void AVXmul64(float *a, float *b, float *c, int ARR_SIZE)
{
    for (int i=0; i < ARR_SIZE ; i+=64){

        __m256 res8 __attribute__(( aligned(32))) = _mm256_mul_ps(_mm256_load_ps(&a[i]),_mm256_load_ps(&b[i]));

        _mm256_store_ps(&c[i],res8);

        __m256 res9 __attribute__(( aligned(32))) = _mm256_mul_ps(_mm256_load_ps(&a[i+8]),_mm256_load_ps(&b[i+8])); // adding 8 values of array a and b

        _mm256_store_ps(&c[i+8],res9);

        __m256 res10 __attribute__(( aligned(32))) = _mm256_mul_ps(_mm256_load_ps(&a[i+16]),_mm256_load_ps(&b[i+16])); // adding 8 values of array a and b

        _mm256_store_ps(&c[i+16],res10);

        __m256 res11 __attribute__(( aligned(32))) = _mm256_mul_ps(_mm256_load_ps(&a[i+24]),_mm256_load_ps(&b[i+24])); // adding 8 values of array a and b

        _mm256_store_ps(&c[i+24],res11);

        __m256 res12 __attribute__(( aligned(32))) = _mm256_mul_ps(_mm256_load_ps(&a[i+32]),_mm256_load_ps(&b[i+32])); // adding 8 values of array a and b

        _mm256_store_ps(&c[i+32],res12);

        __m256 res13 __attribute__(( aligned(32))) = _mm256_mul_ps(_mm256_load_ps(&a[i+40]),_mm256_load_ps(&b[i+40])); // adding 8 values of array a and b

        _mm256_store_ps(&c[i+40],res13);

        __m256 res14 __attribute__(( aligned(32))) = _mm256_mul_ps(_mm256_load_ps(&a[i+48]),_mm256_load_ps(&b[i+48])); // adding 8 values of array a and b

        _mm256_store_ps(&c[i+48],res14);

        __m256 res15 __attribute__(( aligned(32))) = _mm256_mul_ps(_mm256_load_ps(&a[i+56]),_mm256_load_ps(&b[i+56])); // adding 8 values of array a and b

        _mm256_store_ps(&c[i+56],res15);

    }
}

inline void AVXsum(float *a, float *b, float *c, int ARR_SIZE)
{
    for (int i=0; i < ARR_SIZE ; i+=8){

        __m256 vecA __attribute__(( aligned(32))) = _mm256_load_ps(&a[i]); // loading 8 values starting from the address of "i"th value of array a

         __m256 vecB __attribute__(( aligned(32))) = _mm256_load_ps(&b[i]); // loading 8 values starting from the address of "i"th value of array b

         __m256 res __attribute__(( aligned(32))) = _mm256_add_ps(vecA,vecB); // adding 8 values of array a and b

         _mm256_store_ps(&c[i],res); // storing the value in the "i"th address of another array c

    }
}

inline void AVXmul(float *a, float *b, float *c, int ARR_SIZE)
{
    for (int i=0; i < ARR_SIZE ; i+=8){

        __m256 vecA __attribute__(( aligned(32))) = _mm256_load_ps(&a[i]); // loading 8 values starting from the address of "i"th value of array a

         __m256 vecB __attribute__(( aligned(32))) = _mm256_load_ps(&b[i]); // loading 8 values starting from the address of "i"th value of array b

         __m256 res __attribute__(( aligned(32))) = _mm256_mul_ps(vecA,vecB); // adding 8 values of array a and b

         _mm256_store_ps(&c[i],res); // storing the value in the "i"th address of another array c

    }

}


inline void AVXsumMod(float *a, float *b, float *c, int ARR_SIZE)
{
    for (int i=0; i < ARR_SIZE ; i+=8){

        __m256 res __attribute__(( aligned(32))) = _mm256_add_ps(_mm256_load_ps(&a[i]),_mm256_load_ps(&b[i])); // adding 8 values of array a and b

        _mm256_store_ps(&c[i],res); // storing the value in the "i"th address of another array c

    }
}

inline void AVXmulMod(float *a, float *b, float *c, int ARR_SIZE)
{
    for (int i=0; i < ARR_SIZE ; i+=8){

        __m256 res __attribute__(( aligned(32))) = _mm256_mul_ps(_mm256_load_ps(&a[i]),_mm256_load_ps(&b[i]));

        _mm256_store_ps(&c[i],res);

    }
}

inline void Normalsum(float *a, float *b, float *c, int ARR_SIZE)
{
    float add;
    for (int i=0; i < ARR_SIZE ; i++){

        add = a[i] + b[i];
        c[i] = add;
    }
}

inline void Normalmul(float *a, float *b, float *c, int ARR_SIZE)
{
    float mult;
    for (int i=0; i < ARR_SIZE ; i++){

        mult = a[i] * b[i];
        c[i] = mult;
    }
}


static void isAVX2(){
    #if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
        printf("This is a Windows machine and ");
    #else
        printf("\n");
        printf("This is a Linux machine and ");
    #endif

    if ( can_use_intel_core_4th_gen_features() ){
        printf("this CPU supports AVX2\n\n");
    }
    else{
        printf("this CPU does not support AVX2\n");
    }
}

#define size 10000
#define arrsize size
#define NUM_LOOP 10000

static void AVX64(){

    double  normalsumTime, normalmulTime, avxsum64Time, avxmul64Time;
    struct timespec tStart, tEnd;
    double tTotal , tBest=100000;
    int w =0;// do-while loop counter

    printf("the size of array is: %d \n", size);

    // initialization of array and generating random value as per entered size stated above
    float a[size] __attribute__(( aligned(32)));
    for(int i=0; i<size; i++){
        a[i] = (rand()%100)+1;
    }

    float b[size] __attribute__(( aligned(32)));
    for(int i=0; i<size; i++){
        b[i] = (rand()%100)+1;
    }

    float c[arrsize] __attribute__(( aligned(32)));

    //the function is called and time is calculated
    printf("\nNormal Addition... \n\n");
    do{// this loop repeat the body to record the best time
        clock_gettime(CLOCK_MONOTONIC,&tStart);
        Normalsum((float*)&a, (float*)&b , (float*)&c, arrsize);

        clock_gettime(CLOCK_MONOTONIC,&tEnd);
        tTotal = (tEnd.tv_sec - tStart.tv_sec);
        tTotal += (tEnd.tv_nsec - tStart.tv_nsec) / 1000000000.0;

        if(tTotal<tBest)
            tBest=tTotal;
    } while(w++ < NUM_LOOP);

    normalsumTime = tBest;
    tBest = 100000;
    w=0;

    printf("Normal Multiplication... \n\n");
    do{// this loop repeat the body to record the best time
        clock_gettime(CLOCK_MONOTONIC,&tStart);
        Normalmul((float*)&a, (float*)&b , (float*)&c, arrsize);

        clock_gettime(CLOCK_MONOTONIC,&tEnd);
        tTotal = (tEnd.tv_sec - tStart.tv_sec);
        tTotal += (tEnd.tv_nsec - tStart.tv_nsec) / 1000000000.0;

        if(tTotal<tBest)
            tBest=tTotal;
    } while(w++ < NUM_LOOP);

    normalmulTime = tBest;
    tBest = 100000;
    w=0;

    printf("AVX Addition...\n\n");
    do{// this loop repeat the body to record the best time
        clock_gettime(CLOCK_MONOTONIC,&tStart);
        AVXsum64((float*)&a, (float*)&b , (float*)&c, arrsize);

        clock_gettime(CLOCK_MONOTONIC,&tEnd);
        tTotal = (tEnd.tv_sec - tStart.tv_sec);
        tTotal += (tEnd.tv_nsec - tStart.tv_nsec) / 1000000000.0;

        if(tTotal<tBest)
            tBest=tTotal;
    } while(w++ < NUM_LOOP);

    avxsum64Time = tBest;
    tBest = 100000;
    w=0;

    printf("AVX Multiplication...\n\n");
    do{// this loop repeat the body to record the best time
        clock_gettime(CLOCK_MONOTONIC,&tStart);
        AVXmul64((float*)&a, (float*)&b , (float*)&c, arrsize);

        clock_gettime(CLOCK_MONOTONIC,&tEnd);
        tTotal = (tEnd.tv_sec - tStart.tv_sec);
        tTotal += (tEnd.tv_nsec - tStart.tv_nsec) / 1000000000.0;

        if(tTotal<tBest)
            tBest=tTotal;
    } while(w++ < NUM_LOOP);

    avxmul64Time = tBest;

    //printing the output
    printf("Printing the below output when taking 64 values in a per loop cycle(i = i+64) but loading & adding/Multiplying 8 values in a 256 bit register at the same time within single instruction");
    printf("\n");
    printf("\n");
    printf("Normal Sum took %lf s\n" , normalsumTime);
    printf("Normal Mul took %lf s\n",  normalmulTime);
    printf("AVX Sum took %lf s \n", avxsum64Time);
    printf("AVX Mul took %lf s\n", avxmul64Time);
    printf("Sum SpeedUP AVX= %lf ", normalsumTime / avxsum64Time );
    printf("Mul SpeedUP AVX= %lf \n", normalmulTime / avxmul64Time );
    printf( "===========================\n");
    printf( "===========================\n");

}

static void AVX(){

    double  normalsumTime, normalmulTime, avxsumTime, avxmulTime;
    struct timespec tStart, tEnd;
    double tTotal , tBest=100000;
    int w =0;// do-while loop counter

    printf("the size of array is: %d \n", size);

    // initialization of array and generating random value as per entered size stated above
    float a[size] __attribute__(( aligned(32)));
    for(int i=0; i<size; i++){
        a[i] = (rand()%100)+1;
    }

    float b[size] __attribute__(( aligned(32)));
    for(int i=0; i<size; i++){
        b[i] = (rand()%100)+1;
    }

    float c[arrsize] __attribute__(( aligned(32)));

    //the function is called and time is calculated
    printf("\nNormal Addition... \n\n");
    do{// this loop repeat the body to record the best time
        clock_gettime(CLOCK_MONOTONIC,&tStart);
        Normalsum((float*)&a, (float*)&b , (float*)&c, arrsize);

        clock_gettime(CLOCK_MONOTONIC,&tEnd);
        tTotal = (tEnd.tv_sec - tStart.tv_sec);
        tTotal += (tEnd.tv_nsec - tStart.tv_nsec) / 1000000000.0;

        if(tTotal<tBest)
            tBest=tTotal;
    } while(w++ < NUM_LOOP);

    normalsumTime = tBest;
    tBest = 100000;
    w=0;

    printf("Normal Multiplication... \n\n");
    do{// this loop repeat the body to record the best time
        clock_gettime(CLOCK_MONOTONIC,&tStart);
        Normalmul((float*)&a, (float*)&b , (float*)&c, arrsize);

        clock_gettime(CLOCK_MONOTONIC,&tEnd);
        tTotal = (tEnd.tv_sec - tStart.tv_sec);
        tTotal += (tEnd.tv_nsec - tStart.tv_nsec) / 1000000000.0;

        if(tTotal<tBest)
            tBest=tTotal;
    } while(w++ < NUM_LOOP);

    normalmulTime = tBest;
    tBest = 100000;
    w=0;

    printf("AVX Addition...\n\n");
    do{// this loop repeat the body to record the best time
        clock_gettime(CLOCK_MONOTONIC,&tStart);
        AVXsum((float*)&a, (float*)&b , (float*)&c, arrsize);

        clock_gettime(CLOCK_MONOTONIC,&tEnd);
        tTotal = (tEnd.tv_sec - tStart.tv_sec);
        tTotal += (tEnd.tv_nsec - tStart.tv_nsec) / 1000000000.0;

        if(tTotal<tBest)
            tBest=tTotal;
    } while(w++ < NUM_LOOP);

    avxsumTime = tBest;
    tBest = 100000;
    w=0;

    printf("AVX Multiplication...\n\n");
    do{// this loop repeat the body to record the best time
        clock_gettime(CLOCK_MONOTONIC,&tStart);
        AVXmul((float*)&a, (float*)&b , (float*)&c, arrsize);

        clock_gettime(CLOCK_MONOTONIC,&tEnd);
        tTotal = (tEnd.tv_sec - tStart.tv_sec);
        tTotal += (tEnd.tv_nsec - tStart.tv_nsec) / 1000000000.0;

        if(tTotal<tBest)
            tBest=tTotal;
    } while(w++ < NUM_LOOP);

    avxmulTime = tBest;

    //printing the output
    printf("Printing the below output when loading & adding / Multiplying 8 values from array a & b with several instruction");
    printf("\n");
    printf("\n");
    printf("Normal Sum took %lf s\n" , normalsumTime);
    printf("Normal Mul took %lf s\n",  normalmulTime);
    printf("AVX Sum took %lf s \n", avxsumTime);
    printf("AVX Mul took %lf s\n", avxmulTime);
    printf("Sum SpeedUP AVX= %lf ", normalsumTime / avxsumTime );
    printf("Mul SpeedUP AVX= %lf \n", normalmulTime / avxmulTime );
    printf( "===========================\n");
    printf( "===========================\n");

}

static void AVX_Mod(){

    double  normalsumTime, normalmulTime, avxsumModTime, avxmulModTime;
    struct timespec tStart, tEnd;
    double tTotal , tBest=100000;
    int w =0;// do-while loop counter

    printf("the size of array is: %d \n", size);

    // initialization of array and generating random value as per entered size stated above
    float a[size] __attribute__(( aligned(32)));
    for(int i=0; i<size; i++){
        a[i] = (rand()%100)+1;
    }

    float b[size] __attribute__(( aligned(32)));
    for(int i=0; i<size; i++){
        b[i] = (rand()%100)+1;
    }
    float c[arrsize] __attribute__(( aligned(32)));

    printf("\nNormal Addition... \n\n");
    do{// this loop repeat the body to record the best time
        clock_gettime(CLOCK_MONOTONIC,&tStart);
        Normalsum((float*)&a, (float*)&b , (float*)&c, arrsize);

        clock_gettime(CLOCK_MONOTONIC,&tEnd);
        tTotal = (tEnd.tv_sec - tStart.tv_sec);
        tTotal += (tEnd.tv_nsec - tStart.tv_nsec) / 1000000000.0;

        if(tTotal<tBest)
            tBest=tTotal;
    } while(w++ < NUM_LOOP);

    normalsumTime = tBest;
    tBest = 100000;
    w=0;

    printf("Normal Multiplication... \n\n");
    do{// this loop repeat the body to record the best time
        clock_gettime(CLOCK_MONOTONIC,&tStart);
        Normalmul((float*)&a, (float*)&b , (float*)&c, arrsize);

        clock_gettime(CLOCK_MONOTONIC,&tEnd);
        tTotal = (tEnd.tv_sec - tStart.tv_sec);
        tTotal += (tEnd.tv_nsec - tStart.tv_nsec) / 1000000000.0;

        if(tTotal<tBest)
            tBest=tTotal;
    } while(w++ < NUM_LOOP);

    normalmulTime = tBest;
    tBest = 100000;
    w=0;

    printf("AVX Addition...\n\n");
    do{// this loop repeat the body to record the best time
        clock_gettime(CLOCK_MONOTONIC,&tStart);
        AVXsumMod((float*)&a, (float*)&b , (float*)&c, arrsize);

        clock_gettime(CLOCK_MONOTONIC,&tEnd);
        tTotal = (tEnd.tv_sec - tStart.tv_sec);
        tTotal += (tEnd.tv_nsec - tStart.tv_nsec) / 1000000000.0;

        if(tTotal<tBest)
            tBest=tTotal;
    } while(w++ < NUM_LOOP);

    avxsumModTime = tBest;
    tBest = 100000;
    w=0;

    printf("AVX Multiplication...\n\n");
    do{// this loop repeat the body to record the best time
        clock_gettime(CLOCK_MONOTONIC,&tStart);
        AVXmulMod((float*)&a, (float*)&b , (float*)&c, arrsize);

        clock_gettime(CLOCK_MONOTONIC,&tEnd);
        tTotal = (tEnd.tv_sec - tStart.tv_sec);
        tTotal += (tEnd.tv_nsec - tStart.tv_nsec) / 1000000000.0;

        if(tTotal<tBest)
            tBest=tTotal;
    } while(w++ < NUM_LOOP);

    avxmulModTime = tBest;

    //printing the output
    printf("Printing the below output when loading & adding / Multiplying 8 values at the same time from array a & b within single instruction");
    printf("\n");
    printf("\n");
    printf("Normal Sum took %lf s\n" , normalsumTime);
    printf("Normal Mul took %lf s\n",  normalmulTime);
    printf("AVX Sum took %lf s \n", avxsumModTime);
    printf("AVX Mul took %lf s\n", avxmulModTime);
    printf("Sum SpeedUP AVX= %lf ", normalsumTime / avxsumModTime );
    printf("Mul SpeedUP AVX= %lf \n", normalmulTime / avxmulModTime );
    printf( "===========================\n");
    printf( "===========================\n");

}




int main(int argc, char** argv){

    isAVX2();
    AVX();
    AVX_Mod();
    AVX64();

    return 1;
}




