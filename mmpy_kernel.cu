// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
#include "mytypes.h"
using namespace std;

#include <stdio.h>

#define BLOCKTILE_M 96
#define BLOCKTILE_N 64
#define BLOCKTILE_K 32
#define THREADTILE_M (BLOCKTILE_M/BLOCKDIM_Y)
#define THREADTILE_N (BLOCKTILE_N/BLOCKDIM_X)

#define BY BLOCKDIM_Y
#define BX BLOCKDIM_X
#define BK BLOCKTILE_K
#define BM BLOCKTILE_M
#define BN BLOCKTILE_N
#define TM THREADTILE_M
#define TN THREADTILE_N
#define get_mat(mat,N,i,j)((i<N)&&(j<N)?mat[i*N+j]:0)

//
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
//


__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
    __shared__ _DOUBLE_ Ab[BM][BK];
    __shared__ _DOUBLE_ Bb[BK][BN];
    _DOUBLE_ frag_a[TM];
    _DOUBLE_ frag_b[TN];
    _DOUBLE_ c[TM][TN]={0};


    int ty = threadIdx.y, tx=threadIdx.x;
    int by = blockIdx.y, bx=blockIdx.x;

    int I =  by*BM + ty;
    int J =  bx*BN + tx;
    
    int I0 = by * BLOCK_SIZE_M;
    int J0 = bx * BLOCK_SIZE_N;

    #pragma unroll
    for(int K=0;K<N;K+=BK){
        #pragma unroll
        for(int i=0;i<BM;i+=BY){
            #pragma unroll
            for(int j=0;j<BK;j+=BX){
                Ab[ty+i][tx+j]=get_mat(A,N,I+i,K+tx+j);
            }
        }
        #pragma unroll
        for(int i=0;i<BK;i+=BY){
            #pragma unroll
            for(int j=0;j<BN;j+=BX){
                Bb[ty+i][tx+j]=get_mat(B,N,K+ty+i,J+j);
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
__global__ void matMul_shared(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

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
    for (int kk=0; kk<N/TW; ++kk){
        As[ty][tx] = A[I*N + kk*TW+tx];
        Bs[ty][tx] = B[(kk*TW+ty)*N + J];
        __syncthreads();
        for (int k=0; k<TW; ++k)
            Cij+= As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[I*N + J] = Cij;
}

__global__ void matMul_naive(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

    int I =  blockIdx.y*blockDim.y + threadIdx.y;
    int J =  blockIdx.x*blockDim.x + threadIdx.x;

    if((I < N) && (J < N)){
        _DOUBLE_ _c = 0;
        for (unsigned int k = 0; k < N; ++k) {
            _DOUBLE_ a = A[I * N + k];
            _DOUBLE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
}



