
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "random.h"

// uncomment this #define if you want diagnostic output
//#define     DEBUG         0

#define     num_trials    1000000 // number of x values
#define     num_buckets   50         // number of buckets in hitogram
static long xlow        = 0.0;      // low end of x range
static long xhi         = 100.0;    // High end of x range

int main (){

    double x[num_trials];     // array used to assign counters in the historgram
    long   hist[num_buckets]; // the histogram
    double bucket_width;      // the width of each bucket in the histogram
    double time, time_critical, time_lock, time_red;


    seed(xlow, xhi);  // seed the random number generator over range of x
    bucket_width = (xhi-xlow)/(double)num_buckets;

    // fill the array
    for(int i=0;i<num_trials;i++)
        x[i] = drandom();

    ////////////////////////////////////////////////////////////////
    // Assign x values to the right histogram bucket -- sequential
    ////////////////////////////////////////////////////////////////

    // initialize the histogram -> this can be turned into a function
    for(int i=0;i<num_buckets;i++)
        hist[i] = 0;

    // Assign x values to the right historgram bucket
    time = omp_get_wtime();
    for(int i=0;i<num_trials;i++){

        long ival = (long) (x[i] - xlow)/bucket_width;

        hist[ival]++;

        #ifdef DEBUG
        printf("i = %d,  xi = %f, ival = %d\n",i,(float)x[i], ival);
        #endif

    }

    time = omp_get_wtime() - time;

    // compute statistics ... ave, std-dev for whole histogram and quartiles
    // -> this can be turned into a function
    double sumh=0.0, sumhsq=0.0, ave, std_dev;
    for(int i=0;i<num_buckets;i++){
        sumh   += (double) hist[i];
        sumhsq += (double) hist[i]*hist[i];
    }

    ave     = sumh/num_buckets;
    std_dev = sqrt(sumhsq - sumh*sumh/(double)num_buckets);

    printf(" Sequential histogram for %d buckets of %d values\n",num_buckets, num_trials);
    printf(" ave = %f, std_dev = %f\n",(float)ave, (float)std_dev);
    printf(" in %f seconds\n",(float)time);

    ////////////////////////////////////////////////////////////////
    // Assign x values to the right histogram bucket -- critical
    ////////////////////////////////////////////////////////////////

    time_critical = omp_get_wtime();
    double sumh_critical=0.0, sumhsq_critical=0.0, ave_critical, std_dev_critical;
    #pragma omp parallel for shared(sumh, sumhsq)
    for(int i=0;i<num_buckets;i++){
        #pragma omp critical
        {
            sumh_critical   += (double) hist[i];
            sumhsq_critical += (double) hist[i]*hist[i];
        }
    }

    ave_critical     = sumh_critical/num_buckets;
    std_dev_critical = sqrt(sumhsq_critical - sumh_critical*sumh_critical/(double)num_buckets);
    time_critical = omp_get_wtime() - time_critical;


    printf(" par with critical  histogram for %d buckets of %d values\n",num_buckets, num_trials);
    printf(" ave = %f, std_dev = %f\n",(float)ave_critical, (float)std_dev_critical);
    printf(" in %f seconds\n",(float)time_critical);

    ////////////////////////////////////////////////////////////////
    // Assign x values to the right histogram bucket -- locks
    ////////////////////////////////////////////////////////////////
    
    time_lock = omp_get_wtime();
    double sumh_lock=0.0, sumhsq_lock=0.0, ave_lock, std_dev_lock;
    omp_lock_t lock;
    omp_init_lock(&lock);
    #pragma omp parallel for shared(sumh,sumhsq)
    for(int i=0;i<num_buckets;i++){
        omp_set_lock(&lock);
        sumh_lock   += (double) hist[i];
        sumhsq_lock += (double) hist[i]*hist[i];
        omp_unset_lock(&lock);
    }

    ave_lock    = sumh_lock/num_buckets;
    std_dev_lock = sqrt(sumhsq_lock - sumh_lock*sumh_lock/(double)num_buckets);
    time_lock = omp_get_wtime() - time_lock;


    printf(" par with locks  histogram for %d buckets of %d values\n",num_buckets, num_trials);
    printf(" ave = %f, std_dev = %f\n",(float)ave_lock, (float)std_dev_lock);
    printf(" in %f seconds\n",(float)time_lock);

    ////////////////////////////////////////////////////////////////
    // Assign x values to the right histogram bucket -- reduction
    ////////////////////////////////////////////////////////////////
    
    time_red = omp_get_wtime();
    double sumh_red=0.0, sumhsq_red=0.0, ave_red, std_dev_red;
    #pragma omp parallel for reduction(+:sumh_red,sumhsq_red)
    for(int i=0;i<num_buckets;i++){
        sumh_red   += (double) hist[i];
        sumhsq_red += (double) hist[i]*hist[i];
    }

    ave_red     = sumh_red/num_buckets;
    std_dev_red = sqrt(sumhsq_red - sumh_red*sumh_red/(double)num_buckets);
    time_red = omp_get_wtime() - time_red;

    printf(" par with reduction histogram for %d buckets of %d values\n",num_buckets, num_trials);
    printf(" ave = %f, std_dev = %f\n",(float)ave_red, (float)std_dev_red);
    printf(" in %f seconds\n",(float)time_red);


    return 0;
}
	  
