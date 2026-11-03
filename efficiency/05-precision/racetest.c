// 8 threads call lp_gemm_bf16 concurrently on identical inputs (different C buffers); compare with a single-thread reference.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <dlfcn.h>
typedef void (*g_t)(int,int,int,int,int,int,float,const float*,int,const float*,int,float,float*,int);
static g_t g; static float *A,*B,*ref; static int M=512,N=1024,K=256; static int bad[8];
static void *work(void *p){ long id=(long)p; float *C=malloc(4L*M*N);
  for(int it=0;it<300;it++){ g(101,111,111,M,N,K,1,A,K,B,N,0,C,N); if(memcmp(C,ref,4L*M*N)) bad[id]++; } return 0; }
int main(){ void*h=dlopen("./libgemmlp.so",RTLD_NOW); int(*init)(const char*)=dlsym(h,"lp_init");
  printf("init %d\n",init("/home/box/lisp-charmodel/venv/lib/libmkl_rt.so.3")); g=(g_t)dlsym(h,"lp_gemm_bf16");
  A=malloc(4L*M*K);B=malloc(4L*K*N);ref=malloc(4L*M*N); for(long i=0;i<M*K;i++)A[i]=rand()/(float)RAND_MAX-.5f; for(long i=0;i<K*N;i++)B[i]=rand()/(float)RAND_MAX-.5f;
  g(101,111,111,M,N,K,1,A,K,B,N,0,ref,N);
  pthread_t t[8]; for(long i=0;i<8;i++)pthread_create(&t[i],0,work,(void*)i); for(int i=0;i<8;i++)pthread_join(t[i],0);
  for(int i=0;i<8;i++)printf("thread %d mismatches %d/300\n",i,bad[i]); }
