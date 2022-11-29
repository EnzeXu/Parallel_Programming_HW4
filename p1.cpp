#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <cassert>
#include <mpi.h>
using namespace std;

const bool DEBUG = false;
const bool DEBUG_DEFAULT = false;
const bool CHECK_CORRECTNESS = true;

// initialize matrix and vectors (A is mxn, x is xn-vec)
void init_rand(double* a, int m, int n, double* x, int xn);
void init_rand_identity(double* a, int m, int n, double* x, int xn, int row_start, int col_start);
// local matvec: y = y+A*x, where A is m x n
void local_gemv(double* A, double* x, double* y, int m, int n);

int main(int argc, char** argv) {

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    int nProcs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(rank*12345);

    // Read dimensions and processor grid from command line arguments
    if (argc != 5 && argc !=6) {
        cerr << "Usage: ./a.out rows cols pr pc" << endl;
        return 1;
    }
    int identity_flag = 0;
    if (argc == 6) {
        identity_flag = atoi(argv[5]);
    }
    int m, n, pr, pc;
    m  = atoi(argv[1]);
    n  = atoi(argv[2]);
    pr = atoi(argv[3]);
    pc = atoi(argv[4]);
    if (identity_flag) {
        // printf("[identity mode] setting the matrix A as an identity matrix ...\n");
        assert(m == n);
    }
    if(pr * pc != nProcs) {
        cerr << "Processor grid doesn't match number of processors" << endl;
        return 1;
    }
    if(m % pr || n % pc || m % nProcs || n % nProcs) {
        cerr << "Processor grid doesn't divide rows and columns evenly" << endl;
        return 1;
    }

    // Set up row and column communicators
    int ranki = rank % pr; // proc row coordinate
    int rankj = rank / pr; // proc col coordinate
    
    // Create row and column communicators using MPI_Comm_split
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, ranki, ranki, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, rankj, rankj, &col_comm);
    

    // Check row and column communicators and proc coordinates
    int rankichk, rankjchk;
    MPI_Comm_rank(row_comm, &rankjchk);
    MPI_Comm_rank(col_comm, &rankichk);
    if(ranki != rankichk || rankj != rankjchk) {
        cerr << "Processor ranks are not as expected, check row and column communicators" << endl;
        return 1;
    }

    // Initialize matrices and vectors
    int mloc = m / pr;     // number of rows of local matrix
    int nloc = n / pc;     // number of cols of local matrix
    int ydim = m / nProcs; // number of entries of local output vector
    int xdim = n / nProcs; // number of entries of local output vector
    double* Alocal = new double[mloc*nloc];
    double* xlocal = new double[xdim];
    double* ylocal = new double[ydim];
    if (identity_flag) {
        // printf("row start at %d, col start at %d\n", ranki * mloc, rankj * nloc);
        init_rand_identity(Alocal, mloc, nloc, xlocal, xdim, ranki * mloc, rankj * nloc);
    }
    else init_rand(Alocal, mloc, nloc, xlocal, xdim);
    if (DEBUG_DEFAULT) {
        printf("Proc (%d, %d) has initial x_local: x[%d:%d] = ", ranki, rankj, (n / nProcs) * (ranki + rankj * pr), (n / nProcs) * (ranki + rankj * pr + 1));
        for (int i = 0; i < xdim; i++) {
            printf("%lf ", xlocal[i]);
        }
        printf("\n");
    }
    if (DEBUG) {
        printf("Proc (%d, %d) has initial A_local: \n", ranki, rankj);
        for (int i = 0; i < mloc; i++) {
            for (int j = 0; j < nloc; j++) {
                printf("%lf ", Alocal[i+j*mloc]);
            }
            printf("\n");
        }
    }
    memset(ylocal, 0, ydim*sizeof(double));

    // start timer
    double time, start = MPI_Wtime();

    // Communicate input vector entries
    MPI_Status status;

    // for (int i = 0; i < pr; i++) {
    //     if (i != ranki) {
    //         MPI_Send(&xlocal, xdim, MPI_DOUBLE, i, rankj, col_comm);
    //         printf("Proc (%d, %d) send to %d:", ranki, rankj, i);
    //         for (int j = 0; j < xdim; j++) {
    //             printf("%lf ", xlocal[j]);
    //         }
    //         printf("\n");
    //     }
    // }
    // double* xreceive = new double[xdim * pr];
    // for (int i = 0; i < pr; i++) {
    //     if (i != ranki) {
    //         MPI_Recv(xreceive + i * xdim, xdim, MPI_DOUBLE, i, rankj, col_comm, &status);
    //         printf("Proc (%d, %d) receive from %d:\n", ranki, rankj, i);
    //     }
    // }
    // printf("Proc (%d, %d) has xreceive:\n", ranki, rankj);
    // for (int i = 0; i < xdim * pr; i++) {
    //     printf("%lf ", xreceive[i]);
    // }
    // printf("\n");
    double* xreceive = new double[xdim * pr];
    for (int i = 0; i < xdim; i++) {
        xreceive[xdim * ranki + i] = xlocal[i];
    }
    if (DEBUG) {
        printf("Proc (%d, %d) has xreceive [before]: ", ranki, rankj);
        for (int i = 0; i < xdim * pr; i++) {
            printf("%lf ", xreceive[i]);
        }
        printf("\n");
    }
    // for (int i = 0; i < pr; i ++) {
    //     MPI_Bcast(xreceive + xdim * i, xdim, MPI_DOUBLE, i, col_comm);
    // }

    MPI_Allgather(xlocal, xdim, MPI_DOUBLE, xreceive, xdim, MPI_DOUBLE, col_comm);

    // MPI_Bcast(xreceive, xdim * pr, MPI_DOUBLE, ranki, col_comm);
    if (DEBUG) {
        printf("Proc (%d, %d) has xreceive [after]: ", ranki, rankj);
        for (int i = 0; i < xdim * pr; i++) {
            printf("%lf ", xreceive[i]);
        }
        printf("\n");
    }

    // Perform local matvec
    double* local_matvec_result = new double[mloc];
    local_gemv(Alocal, xreceive, local_matvec_result, mloc, nloc);
    if (DEBUG) {
        printf("Proc (%d, %d) has local_matvec_result: ", ranki, rankj);
        for (int i = 0; i < mloc; i++) {
            printf("%lf ", local_matvec_result[i]);
        }
        printf("\n");
    }

    // Communicate output vector entries
    // if (rankj == 0) {
    //     double* sum_result = new double[mloc];
    //     MPI_Reduce(&local_matvec_result, &sum_result, 1, MPI_DOUBLE,MPI_SUM,0,row_comm);
    // }
    // double* sum_result = new double[mloc];
    // for (int i = 0; i < mloc; i++) {
    //     MPI_Reduce(&local_matvec_result[i], &sum_result[i], 1, MPI_DOUBLE, MPI_SUM, 0, row_comm);
    // }

    // MPI_Reduce(&sum, &recv_sum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    // if (DEBUG) {
    //     if (rankj == 0) {
    //         printf("Proc (%d, %d) has sum_result: ", ranki, rankj);
    //         for (int i = 0; i < mloc; i++) {
    //             printf("%lf ", sum_result[i]);
    //         }
    //     }
    //     printf("\n");
    // }
    // Bonus: redistribute the output vector to match input vector

    // double* distribute_result = new double[ydim];

    int *rc = new int[pc];
    for (int i=0; i < pc; i++) {
        rc[i] = ydim;
    }

    // MPI_Reduce_scatter(local_matvec_result, distribute_result, rc, MPI_DOUBLE, MPI_SUM, row_comm);

    MPI_Reduce_scatter(local_matvec_result, ylocal, rc, MPI_DOUBLE, MPI_SUM, row_comm);



    // MPI_Scatter(sum_result, ydim, MPI_DOUBLE, distribute_result, ydim, MPI_DOUBLE, 0, row_comm);
    if (DEBUG_DEFAULT) {
        printf("Proc (%d, %d) has result y_local: y[%d:%d] = ", ranki, rankj, (m / nProcs) * (rankj + ranki * pc), (m / nProcs) * (rankj + ranki * pc + 1));
        for (int i = 0; i < ydim; i++) {
            // printf("%lf ", distribute_result[i]);
            printf("%lf ", ylocal[i]);
        }
        printf("\n");
    }
    // MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    // void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
    // MPI_Comm comm)
    double *x_all = new double[n]; // (n / nProcs) * (ranki + rankj * pr), (n / nProcs) * (ranki + rankj * pr + 1)
    double *y_all = new double[m]; // (m / nProcs) * (rankj + ranki * pc), (m / nProcs) * (rankj + ranki * pc + 1)
    if (CHECK_CORRECTNESS && identity_flag) {
        MPI_Gather(xlocal, xdim, MPI_DOUBLE, x_all, xdim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(ylocal, ydim, MPI_DOUBLE, y_all, ydim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (DEBUG_DEFAULT) {
                printf("x_all: ");
                for (int i = 0; i < n; i ++) {
                    printf("%lf ", x_all[i]);
                }
                printf("\n");
                printf("y_all: ");
                for (int i = 0; i < n; i ++) {
                    printf("%lf ", y_all[i]);
                }
                printf("\n");
            }
            for (int i = 0; i < pr; i++) {
                for (int j = 0; j < pc; j++) {
                    for (int k = 0; k < xdim; k++) {
                        assert(x_all[xdim * (j + i * pc) + k] == y_all[ydim * (i + j * pr) + k]);
                        // printf("x_all[%d] = %lf, y_all[%d] = %lf\n", xdim * (i + j * pr) + k, x_all[xdim * (i + j * pr) + k], ydim * (j + i * pc) + k, y_all[ydim * (j + i * pc) + k]);
                        // printf("x_all[%d] = %lf, y_all[%d] = %lf\n", xdim * (j + i * pc) + k, x_all[xdim * (j + i * pc) + k], ydim * (i + j * pr) + k, y_all[ydim * (i + j * pr) + k]);
                    }
                }
            }
            printf("Correct at case n = m = %d, p = %d\n", n, nProcs);
        }
    }
    
    // MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)

    // if (rank > 0) {
    //     MPI_Send(&xlocal, n / nProcs, MPI_DOUBLE, 0, rankj, col_comm); // MPI_Send(&xlocal, xdim, MPI_DOUBLE, i, rankj, col_comm);
    // } else if (rank == 0) {
    //     double *x = new double[n];
    //     double *y = new double[m];
        
    //     for (int p = 1; p < np; p++) {
    //         MPI_Recv(&recv_sum, 1, MPI_DOUBLE, p, p, MPI_COMM_WORLD, &status); // MPI_Recv(xreceive + i * xdim, xdim, MPI_DOUBLE, i, rankj, col_comm, &status);
    //         sum += recv_sum;
            
    //     }
    //     double pi_avg = sum / double(np);
    //     printf("From %d procs we get pi = %lf\n", np, pi_avg);
    // }
    
    // Stop timer
    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime() - start;

    // Print results for debugging
    if(DEBUG) {
        cout << "Proc (" << ranki << "," << rankj << ") started with x values\n";
        for(int j = 0; j < xdim; j++) {
            cout << xlocal[j] << " ";
        }
        cout << "\nand ended with y values\n";
        for(int i = 0; i < ydim; i++) {
            cout << ylocal[i] << " ";
        }
        cout << endl; // flush now
        for (int i=0; i <mloc; i++) {
            for (int j = 0; j < nloc; j++) {
                printf("%lf ", Alocal[i + j * mloc]);
            }
            printf("\n");
        }
    }

    

    // Print time
    if(!rank) {
        cout << "Time elapsed: " << time << " seconds" << endl;
    }

    // Clean up
    delete [] ylocal;
    delete [] xlocal;
    delete [] Alocal;
    delete [] xreceive;
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
}

void local_gemv(double* a, double* x, double* y, int m, int n) {
    // order for loops to match col-major storage
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            y[i] += a[i+j*m] * x[j];
        }
    }
}

void init_rand(double* a, int m, int n, double* x, int xn) {
    // init matrix
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            a[i+j*m] = rand() % 100;
        }
    }
    // init input vector x
    for(int j = 0; j < xn; j++) {
        x[j] = rand() % 100;
    }
}

void init_rand_identity(double* a, int m, int n, double* x, int xn, int row_start, int col_start) {
    // init matrix
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            if (row_start + i == col_start + j) {
                if (DEBUG) printf("setting [%d, %d] as 1.0\n", row_start + i, col_start + j);
                a[i+j*m] = 1.0;
            }
            else {
                if (DEBUG) printf("setting [%d, %d] as 0.0\n", row_start + i, col_start + j);
                a[i+j*m] = 0.0;
            }
            // a[i+j*m] = rand() % 100;
        }
    }
    // init input vector x
    for(int j = 0; j < xn; j++) {
        x[j] = rand() % 100;
    }
}
