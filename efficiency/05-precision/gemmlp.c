// Low-precision GEMM shim for gpt-eff.lisp: same interface as cblas_sgemm (row-major, fp32 in/out), but the two
// input operands are rounded to bfloat16 (round-to-nearest-even, AVX512-BF16) into thread-local scratch and multiplied with
// MKL cblas_gemm_bf16bf16f32 (AMX tiles on Sapphire Rapids), accumulating in fp32 into the fp32 C (alpha/beta honoured).
// Master weights, gradients, optimizer state and all non-GEMM math stay fp32/fp64 in Lisp.
#include <stdint.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <immintrin.h>
typedef void (*bgemm_t)(int,int,int,int,int,int,float,const uint16_t*,int,const uint16_t*,int,float,float*,int);
typedef void (*sgemm_t)(int,int,int,int,int,int,float,const float*,int,const float*,int,float,float*,int);
static bgemm_t bg; static sgemm_t sg;
static __thread uint16_t *bufA, *bufB; static __thread long capA, capB;
int lp_init(const char *mkl) {
  void *h = dlopen(mkl, RTLD_NOW | RTLD_GLOBAL); if (!h) return -1;
  bg = (bgemm_t)dlsym(h, "cblas_gemm_bf16bf16f32"); sg = (sgemm_t)dlsym(h, "cblas_sgemm");
  void (*st)(int) = (void(*)(int))dlsym(h, "MKL_Set_Num_Threads"); if (st) st(1);
  return bg && sg ? 0 : -2; }
static void conv(const float *x, int rows, int cols, int ld, uint16_t *y) {   // compact rows x cols -> ld' = cols
  for (int r = 0; r < rows; r++) {
    const float *s = x + (long)r * ld; uint16_t *d = y + (long)r * cols; int c = 0;
    for (; c + 32 <= cols; c += 32) {
      __m512bh v = _mm512_cvtne2ps_pbh(_mm512_loadu_ps(s + c + 16), _mm512_loadu_ps(s + c));
      _mm512_storeu_si512((void *)(d + c), (__m512i)v); }
    for (; c < cols; c++) { __m128bh t = _mm_cvtneps_pbh(_mm_set_ss(s[c])); d[c] = ((uint16_t *)&t)[0]; } } }
static uint16_t *grow(uint16_t **b, long *cap, long n) { if (n > *cap) { free(*b); *b = aligned_alloc(64, ((n * 2 + 63) / 64) * 64); *cap = n; } return *b; }
void lp_sgemm_mkl(int o, int ta, int tb, int m, int n, int k, float al, const float *A, int lda, const float *B, int ldb, float be, float *C, int ldc) {
  sg(o, ta, tb, m, n, k, al, A, lda, B, ldb, be, C, ldc); }
void lp_gemm_bf16(int o, int ta, int tb, int m, int n, int k, float al, const float *A, int lda, const float *B, int ldb, float be, float *C, int ldc) {
  int ar = ta == 112 ? k : m, ac = ta == 112 ? m : k, br = tb == 112 ? n : k, bc = tb == 112 ? k : n;
  uint16_t *a = grow(&bufA, &capA, (long)ar * ac), *b = grow(&bufB, &capB, (long)br * bc);
  conv(A, ar, ac, lda, a); conv(B, br, bc, ldb, b);
  bg(o, ta, tb, m, n, k, al, a, ac, b, bc, be, C, ldc); }
