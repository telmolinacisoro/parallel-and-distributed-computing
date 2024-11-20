#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include "cholesky.h"



void cholesky_openmp(int n) {
    // TODO
    int i, j, k, l;
    double** A;
    double** L;
    double** U;
    double** B;
    double tmp;
    double start, end;
    int cnt;
    
    /**
     * 1. Matrix initialization for A, L, U and B
     */
    // The initialisation will not be paralellised because it will, in fact, take longer due to the overhead
    start = omp_get_wtime();
    A = (double **)malloc(n * sizeof(double *));
    L = (double **)malloc(n * sizeof(double *));
    U = (double **)malloc(n * sizeof(double *));
    B = (double **)malloc(n * sizeof(double *));
    
    for(i=0; i<n; i++) {
         A[i] = (double *)malloc(n * sizeof(double));
         L[i] = (double *)malloc(n * sizeof(double));
         U[i] = (double *)malloc(n * sizeof(double));
         B[i] = (double *)malloc(n * sizeof(double));
    }
    
    srand(time(NULL));
    //Distribute iterations of i, each thread with its own copy of j to avoid race condition
    for(i=0; i < n; i++) {
        A[i][i] = rand()%1000+100.0;
        for(j=i+1; j < n; j++) {
            A[i][j] = rand()%100 + 1.0;
            A[j][i] = A[i][j];
        }
    }
    //Combine the two loops and assign chunks of iterations to each thread
    for(i=0; i < n; i++) {
        for(j=0; j < n; j++) {
            L[i][j] = 0.0;
            U[i][j] = 0.0;
        }
    }
    end = omp_get_wtime();
    printf("Initialization: %f\n", end-start);
    
    
    /**
     * 2. Compute Cholesky factorization for U
     */
    start = omp_get_wtime();
    //Pararelise loop i and create private values of j, k and tmp, to avoid race condition
    //#pragma omp parallel for private(j, k, tmp)
    for(i=0; i<n; i++) {
        // Calculate diagonal elements
        tmp = 0.0;
        #pragma omp parallel for shared(U,A) private(k) reduction(+:tmp)
        for(k=0;k<i;k++) {
            tmp += U[k][i]*U[k][i];
        }
        U[i][i] = sqrt(A[i][i]-tmp);
        // Calculate non-diagonal elements
        #pragma omp parallel for shared(U,A) private(j,k) reduction(+:tmp)
        for(j=i+1;j<n;j++) {
            tmp = 0.0;
            for(k=0;k<i;k++){
                tmp+= U[k][j]*U[k][i];
            }
            U[i][j] = (A[j][i] - tmp)/U[i][i];
        }
    }
    end = omp_get_wtime();
    printf("Cholesky: %f\n", end-start);
    
        
    /**
     * 3. Calculate L from U'
     */
    start = omp_get_wtime();

    double block_size = 64;
    for (i = 0; i < n; i += block_size) {
        for (j = 0; j < n; j += block_size) {
            #pragma omp parallel for shared(L, U) private(k, l)
            for (k = i; k < n; k++) {
                for (l = j; l < n; l++) {
                    if (U[k][l] != 0) {
                        L[l][k] = U[k][l];
                    }
                }
            }
        }
    }



    /**
    * 4. Compute B=LU
    */
    start = omp_get_wtime();
    // TODO B=LU
    //paralelise the cambination of i and j loops and create private values for i,j,k. Matrix memory is shared between threads
    double sum;
    #pragma omp parallel for shared(A,L,U,B) private(i,j,k) reduction(+:sum)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            // Calculate (i,j)-th entry of B
            sum = 0.0;
            for (k = 0; k < n; k++) {
                if (k <= i && k <= j) {
                    sum += L[i][k] * U[k][j];
                }
            }
            B[i][j] = sum;
        }
    }
    end = omp_get_wtime();
    printf("B=LU: %f\n", end-start);

    /**
    * 5. Check if all elements of A and B have a difference smaller than 0.001
    */
    start = omp_get_wtime();
    cnt = 0;
    // TODO check if matrices are equal
    //Paralelise the loop and add values of private cnt to the global cnt at the end of the loop
    #pragma omp parallel for private(i, j) shared(A, B) reduction(+:cnt)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (fabs(A[i][j] - B[i][j]) > 0.001) {
                cnt++;
             }
        }
    }
    if (cnt != 0) {
        printf("Matrices are not equal\n");
    } else {
        printf("Matrices are equal\n");
    }
    end = omp_get_wtime();
    printf("A==B?: %f\n", end-start);
}


void cholesky(int n) {
    int i, j, k;
    double** A;
    double** L;
    double** U;
    double** B;
    double tmp;
    double start, end;
    int cnt;
    
    /**
     * 1. Matrix initialization for A, L, U and B
     */
    start = omp_get_wtime();
    A = (double **)malloc(n * sizeof(double *));
    L = (double **)malloc(n * sizeof(double *));
    U = (double **)malloc(n * sizeof(double *));
    B = (double **)malloc(n * sizeof(double *));
    
    for(i=0; i<n; i++) {
         A[i] = (double *)malloc(n * sizeof(double));
         L[i] = (double *)malloc(n * sizeof(double));
         U[i] = (double *)malloc(n * sizeof(double));
         B[i] = (double *)malloc(n * sizeof(double));
    }
    
    srand(time(NULL));
    for(i=0; i < n; i++) {
        A[i][i] = rand()%1000+100.0;
        for(j=i+1; j < n; j++) {
            A[i][j] = rand()%100 + 1.0;
            A[j][i] = A[i][j];
        }
    }

    for(i=0; i < n; i++) {
        for(j=0; j < n; j++) {
            L[i][j] = 0.0;
            U[i][j] = 0.0;
        }
    }
    end = omp_get_wtime();
    printf("Initialization: %f\n", end-start);
    
    
    /**
     * 2. Compute Cholesky factorization for U
     */
    start = omp_get_wtime();
    for(i=0; i<n; i++) {
        // Calculate diagonal elements
        tmp = 0.0;
        for(k=0;k<i;k++) {
            tmp += U[k][i]*U[k][i];
        }
        U[i][i] = sqrt(A[i][i]-tmp);
        // Calculate non-diagonal elements
        
        for(j=i+1;j<n;j++) {
            tmp = 0.0;
            for(k=0;k<i;k++){
                tmp+= U[k][j]*U[k][i];
            }
            U[i][j] = (A[j][i] - tmp)/U[i][i];
        }
    }
    end = omp_get_wtime();
    printf("Cholesky: %f\n", end-start);
    
        

    /**
    * 3. Calculate L from U'
    */
    start = omp_get_wtime();
    
    double block_size = 64;
    for (i = 0; i <n; i+=block_size) {
        for (j = 0; j < n; j+=block_size) {
            for (k = i; (k < i + block_size) && (k < n); k++){
                for (int l = j; (l < j + block_size) && (l <n); l++) {
                    if (U[k][l]!=0) L[l][k] = U[k][l];
                }
            }
        }
    }
    
    /**
     * 4. Compute B=LU
     */
    start = omp_get_wtime();
    // TODO B=LU
    for( i=0; i<n; i++){
        for( j=0; j<=n; j++){
            // iterate over the non-zero entries in the i-th row of L
                for (int k = 0; k <= i && L[i][k] != 0; k++) {
            // if the k-th entry in U is non-zero
                    if (U[k][j] != 0) {
                        // update the (i,j)-th entry of B by adding the product of the k-th entries in L and U
                        B[i][j] += L[i][k] * U[k][j];
            }
        }
        }
    }
    end = omp_get_wtime();
    printf("B=LU: %f\n", end-start);


    /**
     * 5. Check if all elements of A and B have a difference smaller than 0.001
     */
    start = omp_get_wtime();
    cnt=0;
    // TODO check if matrices are equal
    for( i=0; i<n; i++){
        for( j=0; j<n; j++){
            if(fabs(A[i][j]-B[i][j])>0.001){cnt++;}
        }
    }
    if(cnt != 0) {
        printf("Matrices are not equal\n");
    } else {
        printf("Matrices are equal\n");
    }
    end = omp_get_wtime();
    printf("A==B?: %f\n", end-start);
}

