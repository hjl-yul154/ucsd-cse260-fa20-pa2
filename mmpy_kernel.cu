// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
#include "mytypes.h"
using namespace std;

#include <stdio.h>

<<<<<<< HEAD
#define TW 32
__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
=======
#define BLOCK_SIZE_M 96
#define BLOCK_SIZE_N 64
#define BLOCK_SIZE_K 32
#if BLOCK_SIZE_M % BLOCKDIM_Y || BLOCK_SIZE_K % BLOCKDIM_X
#error Use thread block to load block of A
#endif
#if BLOCK_SIZE_K % BLOCKDIM_Y || BLOCK_SIZE_N % BLOCKDIM_X
#error Use thread block to load block of B
#endif
#if BLOCK_SIZE_M % BLOCKDIM_Y || BLOCK_SIZE_N % BLOCKDIM_X
#error Use thread block to compute block of C
#endif
// Number of sub-block of C for each thread
#define X_SUB (BLOCK_SIZE_N / BLOCKDIM_X)
#define Y_SUB (BLOCK_SIZE_M / BLOCKDIM_Y)

#define MAT(mat, N, i, j) (mat[(i)*N + (j)])
#define MAT_PADDED(mat, N, i, j) ((i) < N && (j) < N ? MAT(mat, N, i, j) : 0)
#define A_ELEMENT(i, j) MAT_PADDED(A, N, i, j)
#define B_ELEMENT(i, j) MAT_PADDED(B, N, i, j)
#define C_ELEMENT(i, j) MAT(C, N, i, j)

__global__ void matMul(int N, _DOUBLE_* C, _DOUBLE_* A, _DOUBLE_* B) {
    __shared__ _DOUBLE_ Ab[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ _DOUBLE_ Bb[BLOCK_SIZE_K][BLOCK_SIZE_N];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    _DOUBLE_ c[Y_SUB][X_SUB] = {0};  // Zero initialize the whole array

    // Compute I0,J0 of C
    int I0 = by * BLOCK_SIZE_M;
    int J0 = bx * BLOCK_SIZE_N;

#pragma unroll
    for (int K = 0; K < N; K += BLOCK_SIZE_K) {
#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_M; i += BLOCKDIM_Y) {
#pragma unroll
            for (int j = 0; j < BLOCK_SIZE_K; j += BLOCKDIM_X) {
                Ab[ty + i][tx + j] = A_ELEMENT(I0 + ty + i, K + tx + j);
            }
        }

#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K; i += BLOCKDIM_Y) {
#pragma unroll
            for (int j = 0; j < BLOCK_SIZE_N; j += BLOCKDIM_X) {
                Bb[ty + i][tx + j] = B_ELEMENT(K + ty + i, J0 + tx + j);
            }
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
#pragma unroll
            for (int i = 0; i < Y_SUB; ++i) {
#pragma unroll
                for (int j = 0; j < X_SUB; ++j) {
                    c[i][j] +=
                        Ab[ty + i * BLOCKDIM_Y][k] * Bb[k][tx + j * BLOCKDIM_X];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < Y_SUB; ++i) {
#pragma unroll
        for (int j = 0; j < X_SUB; ++j) {
            if (I0 + ty + i * BLOCKDIM_Y < N && J0 + tx + j * BLOCKDIM_X < N) {
                C_ELEMENT(I0 + ty + i * BLOCKDIM_Y, J0 + tx + j * BLOCKDIM_X) =
                    c[i][j];
            }
        }
    }
}


#define TW 16
__global__ void matMul_1(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
>>>>>>> 1523a3a39ed65c344e1867b471d7d070ae7dc3a5

    //local shared storage
    int TY = blockDim.y;
    int TX = blockDim.x;
//     int TW = blockDim.x;

    __shared__ _DOUBLE_ As[TW][TW];
    __shared__ _DOUBLE_ Bs[TW][TW];
    int ty = threadIdx.y, tx = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;
    int I = by*TW + ty; int J= bx*TW + tx;
    double Cij = 0;
    for (int kk=0; kk<N/TW; kk++){
        As[ty][tx] = A[I*N + kk*TW+tx];
        Bs[ty][tx] = B[(kk*TW+ty)*N + J];
        __syncthreads();
        for (int k=0; k<TW; k++)
            Cij+= As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[I*N + J] = Cij;
}

__global__ void matMul_old(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

    int I =  blockIdx.y*blockDim.y + threadIdx.y;
    int J =  blockIdx.x*blockDim.x + threadIdx.x;

    if((I < N) && (J < N)){
        _DOUBLE_ _c = 0;
        for (unsigned int k = 0; k < N; k++) {
            _DOUBLE_ a = A[I * N + k];
            _DOUBLE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
}
