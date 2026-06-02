#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdatomic.h>
#include "ricklib.h"
// #include <omp.h>

void stokes_collapse(
    unsigned int Nmeasures,
    int freq_per_chan,
    float *visreal,
    float *visimg,
    float *weights,
    float *visreal_stokes,
    float *visimg_stokes,
    float *weights_stokes)
{
    // In this way we select and combine correlations to form Stokes parameters

#if !defined(WEIGHTING_UNIFORM) && !defined(WEIGHTING_BRIGGS)
    float weights_stokes_sum = 0.0;
#endif

#if defined(STOKESI)
    for (unsigned int i = 0; i < (Nmeasures * freq_per_chan); i++)
    {
        visreal_stokes[i] = 0.5 * (visreal[i * 4] + visreal[(i * 4) + 3]);
        visimg_stokes[i] = 0.5 * (visimg[i * 4] + visimg[(i * 4) + 3]);
        weights_stokes[i] = 0.25 * (weights[i * 4] + weights[(i * 4) + 3]);
        #if !defined(WEIGHTING_UNIFORM) && !defined(WEIGHTING_BRIGGS)
            weights_stokes_sum += weights_stokes[i];
        #endif
    }
    // printf("Sum weights Stokes I %f\n", weights_stokes_sum);
#elif defined(STOKESQ)
    for (unsigned int i = 0; i < (Nmeasures * freq_per_chan); i++)
    {
        visreal_stokes[i] = 0.5 * (visreal[i * 4] - visreal[(i * 4) + 3]);
        visimg_stokes[i] = 0.5 * (visimg[i * 4] - visimg[(i * 4) + 3]);
        weights_stokes[i] = weights[i * 4];
        #if !defined(WEIGHTING_UNIFORM) && !defined(WEIGHTING_BRIGGS)
            weights_stokes_sum += weights_stokes[i];
        #endif
    }
#elif defined(STOKESU)
    for (unsigned int i = 0; i < (Nmeasures * freq_per_chan); i++)
    {
        visreal_stokes[i] = 0.5 * (visreal[(i * 4) + 1] + visreal[(i * 4) + 2]);
        visimg_stokes[i] = 0.5 * (visimg[(i * 4) + 1] + visimg[(i * 4) + 2]);
        weights_stokes[i] = weights[i * 4];
        #if !defined(WEIGHTING_UNIFORM) && !defined(WEIGHTING_BRIGGS)
            weights_stokes_sum += weights_stokes[i];
        #endif
    }
#elif defined(STOKESV)
    for (unsigned int i = 0; i < (Nmeasures * freq_per_chan); i++)
    {
        visreal_stokes[i] = 0.5 * (visreal[(i * 4) + 1] - visreal[(i * 4) + 2]);
        visimg_stokes[i] = 0.5 * (visimg[(i * 4) + 1] - visimg[(i * 4) + 2]);
        weights_stokes[i] = 0.5 * (weights[(i * 4) + 1] + weights[(i * 4) + 2] );
        #if !defined(WEIGHTING_UNIFORM) && !defined(WEIGHTING_BRIGGS)
            weights_stokes_sum += weights_stokes[i];
        #endif
    }
#endif

#if !defined(WEIGHTING_UNIFORM) && !defined(WEIGHTING_BRIGGS)
    printf("Sum of all weights with NATURAL weighting, to be used for normalization: %f\n", weights_stokes_sum);
#endif
    free(visreal);
    free(visimg);
    free(weights);
}