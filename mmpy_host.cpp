#include "types.h"
#include "cblas.h"
void
matMulHost(_DOUBLE_  *C, const _DOUBLE_  *A, const _DOUBLE_  *B, unsigned int M, unsigned int N)
{
    const _DOUBLE_ Beta  = 0.0;
    const _DOUBLE_ Alpha = 1.0;
    const int K = N;
    const int LDA = N, LDB = N, LDC = N;
    const enum CBLAS_TRANSPOSE transA = CblasNoTrans;
    const enum CBLAS_TRANSPOSE transB = CblasNoTrans;
    cblas_dgemm( CblasRowMajor, transA, transB, M, N, K,
                 Alpha, A, LDA, B, LDB, Beta, C, LDC );
}

void
reference_dgemm(unsigned int N, _DOUBLE_ Alpha, _DOUBLE_  *A, _DOUBLE_  *B, _DOUBLE_  *C)
{
    const _DOUBLE_ Beta  = 1.0;
    const int M = N, K = N;
    const int LDA = N, LDB = N, LDC = N;
    const enum CBLAS_TRANSPOSE transA = CblasNoTrans;
    const enum CBLAS_TRANSPOSE transB = CblasNoTrans;
    cblas_dgemm( CblasRowMajor, transA, transB, M, N, K,
                 Alpha, A, LDA, B, LDB, Beta, C, LDC );
}
