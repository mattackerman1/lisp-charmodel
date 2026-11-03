// Same inputs, 8 threads; direct library calls with per-thread pre-converted bf16 inputs. Reports mismatch count and max |diff|.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <dlfcn.h>
typedef void (*b_t)(int,int,int,int,int,int,float,const uint16_t*,int,const uint16_t*,int,float,float*,int);
static b_t g; static uint16_t *A,*B; static float *ref; static int M=512,N=1024,K=256; static int bad[8]; static double mx[8];
static void *work(void *p){ long id=(long)p; float *C=malloc(4L*M*N); uint16_t*a=malloc(2L*M*K),*b=malloc(2L*K*N); memcpy(a,A,2L*M*K); memcpy(b,B,2L*K*N);
  for(int it=0;it<300;it++){ g(101,111,111,M,N,K,1,a,K,b,N,0,C,N);
    if(memcmp(C,ref,4L*M*N)){ bad[id]++; for(long i=0;i<(long)M*N;i++){double d=fabs(C[i]-ref[i]); if(d>mx[id])mx[id]=d;} } } return 0; }
int main(int c,char**v){ void*h=dlopen(v[1],RTLD_NOW); g=(b_t)dlsym(h,v[2]);
  void(*st)(int)=dlsym(h,"MKL_Set_Num_Threads"); if(!st) st=dlsym(h,"openblas_set_num_threads"); st(1);
  A=malloc(2L*M*K);B=malloc(2L*K*N);ref=malloc(4L*M*N); for(long i=0;i<M*K;i++)A[i]=0x3f00+(rand()&0xff); for(long i=0;i<K*N;i++)B[i]=0xbf00+(rand()&0x1ff);
  g(101,111,111,M,N,K,1,A,K,B,N,0,ref,N);
  pthread_t t[8]; for(long i=0;i<8;i++)pthread_create(&t[i],0,work,(void*)i); for(int i=0;i<8;i++)pthread_join(t[i],0);
  for(int i=0;i<8;i++)printf("%s thread %d mismatches %d/300 max|diff| %.3g (|ref|~%.3g)\n",v[2],i,bad[i],mx[i],fabs(ref[0])); }
