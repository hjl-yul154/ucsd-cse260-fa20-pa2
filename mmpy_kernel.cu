// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
#include "mytypes.h"
using namespace std;

#include <stdio.h>

#define get_mat(mat, N, i, j)((i) < N && (j) < N ? mat[(i)*N+(j)] : 0)

__global__ void matMul_cutlass(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
    __shared__ _DOUBLE_ Ab[BM][BK];
    __shared__ _DOUBLE_ Bb[BK][BN];
    _DOUBLE_ frag_a[TM];
    _DOUBLE_ frag_b[TN];
    _DOUBLE_ Cb[TM][TN]={0};


    int ty0 = threadIdx.y, tx0=threadIdx.x;
    int by = blockIdx.y, bx=blockIdx.x;
    
    int ty = ((ty0*BY+tx0)/32)/(BX/WX)*WY+((ty0*BY+tx0)%32)/WX;
    int tx = ((ty0*BY+tx0)/32)%(BX/WX)*WX+((ty0*BY+tx0)%32)%WX;

    int I =  by*BM + ty;
    int J =  bx*BN + tx;


    #pragma unroll
    for(int K=0; K<N; K+=BK){
        #pragma unroll
        for(int i=0;i<BM;i+=BY){
            #pragma unroll
            for(int j=0;j<BK;j+=BX){
                Ab[ty + i][tx + j] = get_mat(A,N,I + i, K + tx + j);
            }
        }
        #pragma unroll
        for(int i=0;i<BK;i+=BY){
            #pragma unroll
            for(int j=0;j<BN;j+=BX){
                Bb[ty + i][tx + j]=get_mat(B,N,K+ty+i,J+j);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int k=0;k<BK;++k){
            #pragma unroll
            for (int i=0;i<TM;++i){
                frag_a[i]=Ab[ty+BY*i][k];
            }
            #pragma unroll
            for (int j=0;j<TN;++j){
                frag_b[j]=Bb[k][tx+BX*j];
            }

            #pragma unroll
            for (int i=0;i<TM;++i){
                #pragma unroll
                for (int j=0;j<TN;++j){
                    Cb[i][j]+=frag_a[i]*frag_b[j];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<TM;++i){
        #pragma unroll
        for(int j=0;j<TN;++j){
            if(I+i*BY<N && J+j*BX<N){
                C[(I+BY*i)*N+J+BX*j]=Cb[i][j];
            }
        }
    }

}

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
    __shared__ _DOUBLE_ Ab[BM][BK];
    __shared__ _DOUBLE_ Bb[BK][BN];
    _DOUBLE_ frag_a[TM];
    _DOUBLE_ frag_b[TN];
    _DOUBLE_ Cb[TM][TN]={0};


    int ty = threadIdx.y, tx=threadIdx.x;
    int by = blockIdx.y, bx=blockIdx.x;

    int I =  by*BM + ty;
    int J =  bx*BN + tx;

    #pragma unroll
    for(int K=0; K<N; K+=BK){
        #pragma unroll
        for(int i=0;i<BM;i+=BY){
            #pragma unroll
            for(int j=0;j<BK;j+=BX){
                Ab[ty + i][tx + j] = get_mat(A,N,I + i, K + tx + j);
            }
        }
        #pragma unroll
        for(int i=0;i<BK;i+=BY){
            #pragma unroll
            for(int j=0;j<BN;j+=BX){
                Bb[ty + i][tx + j]=get_mat(B,N,K+ty+i,J+j);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int k=0;k<BK;k++){
            #pragma unroll
            for (int i=0;i<TM;i++){
                frag_a[i]=Ab[ty+BY*i][k];
            }
            #pragma unroll
            for (int j=0;j<TN;j++){
                frag_b[j]=Bb[k][tx+BX*j];
            }

            #pragma unroll
            for (int i=0;i<TM;i++){
                #pragma unroll
                for (int j=0;j<TN;j++){
                    Cb[i][j]+=frag_a[i]*frag_b[j];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<TM;i++){
        #pragma unroll
        for(int j=0;j<TN;j++){
            if(I+i*BY<N && J+j*BX<N){
                C[(I+BY*i)*N+J+BX*j]=Cb[i][j];
            }
        }
    }

}

__global__ void matMul_shared(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

    //local shared storage
//     int TW = blockDim.x;

    __shared__ _DOUBLE_ Ab[BY][BK];
    __shared__ _DOUBLE_ Bb[BK][BX];


    int ty = threadIdx.y, tx=threadIdx.x;
    int by = blockIdx.y, bx=blockIdx.x;

    int I =  by*BY + ty;
    int J =  bx*BX + tx;

    _DOUBLE_ Cij = 0;

    for(int K=0; K<N; K+=BK){
        Ab[ty][tx] = get_mat(A,N,I, K + tx);
        Bb[ty][tx]=get_mat(B,N,K+ty,J);
        __syncthreads();
        for (int k=0; k<BK; k++){
            Cij+= Ab[ty][k] * Bb[k][tx];
        }
        __syncthreads();
    }
    C[I*N + J] = Cij;
}


__global__ void matMul_naive(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

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
