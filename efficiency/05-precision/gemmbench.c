// Single-thread GEMM benchmark on the shapes of one training worker (R = 8 seqs x 64 tokens = 512 rows, d=256, V=2048).
// fp32 sgemm: system OpenBLAS, MKL; bf16 (fp32 accumulate): MKL cblas_gemm_bf16bf16f32, OpenBLAS cblas_sbgemm (if lib given).
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>
#include <time.h>
typedef void (*sgemm_t)(int,int,int,int,int,int,float,const float*,int,const float*,int,float,float*,int);
typedef void (*bgemm_t)(int,int,int,int,int,int,float,const uint16_t*,int,const uint16_t*,int,float,float*,int);
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static void tobf16(const float*x,uint16_t*y,long n){for(long i=0;i<n;i++){uint32_t u;memcpy(&u,&x[i],4);u+=0x7fff+((u>>16)&1);y[i]=u>>16;}}
typedef struct{const char*name;int ta,tb,m,n,k;}shape;
int main(int argc,char**argv){
  void*ob=dlopen("/usr/lib/x86_64-linux-gnu/libopenblas.so.0",RTLD_NOW|RTLD_LOCAL);
  void*mk=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
  void*ob16=argc>2?dlopen(argv[2],RTLD_NOW|RTLD_LOCAL):NULL;
  ((void(*)(int))dlsym(ob,"openblas_set_num_threads"))(1);
  if(mk) ((void(*)(int))dlsym(mk,"MKL_Set_Num_Threads"))(1);
  if(ob16) ((void(*)(int))dlsym(ob16,"openblas_set_num_threads"))(1);
  sgemm_t s_ob=(sgemm_t)dlsym(ob,"cblas_sgemm"), s_mk=mk?(sgemm_t)dlsym(mk,"cblas_sgemm"):0, s_ob16=ob16?(sgemm_t)dlsym(ob16,"cblas_sgemm"):0;
  bgemm_t b_mk=mk?(bgemm_t)dlsym(mk,"cblas_gemm_bf16bf16f32"):0, b_ob=ob16?(bgemm_t)dlsym(ob16,"cblas_sbgemm"):0;
  printf("syms: mkl sgemm %p bf16 %p | ob16 sgemm %p sbgemm %p\n",(void*)s_mk,(void*)b_mk,(void*)s_ob16,(void*)b_ob);
  shape S[]={{"fwd qkv  X.W   ",0,0,512,768,256},{"fwd proj X.W   ",0,0,512,256,256},{"fwd fc   X.W   ",0,0,512,1024,256},
             {"fwd fc2  X.W   ",0,0,512,256,1024},{"head  X.Wte^T  ",0,1,512,2048,256},{"bwd gW fc X^T.dY",1,0,256,1024,512},
             {"bwd dX fc dY.W^T",0,1,512,256,1024},{"bwd gWte P^T.X ",1,0,2048,256,512},{"attn 64x64x32  ",0,1,64,64,32}};
  int ns=sizeof(S)/sizeof(S[0]); double tot[6]={0};
  printf("%-17s %9s %9s %9s %9s %9s %9s  (GFLOP/s; bf16+cv = incl. fp32->bf16 conversion of both inputs)\n","shape","OB-f32","MKL-f32","OB16-f32","MKL-bf16","MKL-bf16+cv","OB-sbgemm");
  for(int s=0;s<ns;s++){
    shape h=S[s]; long na=(long)h.m*h.k, nb=(long)h.k*h.n, nc=(long)h.m*h.n;
    float*A=malloc(4*na),*B=malloc(4*nb),*C=malloc(4*nc); uint16_t*Ab=malloc(2*na),*Bb=malloc(2*nb);
    for(long i=0;i<na;i++)A[i]=rand()/(float)RAND_MAX-0.5f; for(long i=0;i<nb;i++)B[i]=rand()/(float)RAND_MAX-0.5f;
    tobf16(A,Ab,na); tobf16(B,Bb,nb);
    int lda=h.ta?h.m:h.k, ldb=h.tb?h.k:h.n; int TA=h.ta?112:111, TB=h.tb?112:111;
    double fl=2.0*h.m*h.n*h.k; int reps=(int)(3e9/fl)+3; double g[6]={0};
    for(int v=0;v<6;v++){
      if((v==1||v==3||v==4)&&!mk) continue; if((v==2&&!s_ob16)||(v==5&&!b_ob)) continue; if(v==3&&!b_mk) continue;
      for(int w=0;w<2;w++){ double t0=now();
        for(int r=0;r<reps;r++){
          if(v==0) s_ob(101,TA,TB,h.m,h.n,h.k,1,A,lda,B,ldb,0,C,h.n);
          if(v==1) s_mk(101,TA,TB,h.m,h.n,h.k,1,A,lda,B,ldb,0,C,h.n);
          if(v==2) s_ob16(101,TA,TB,h.m,h.n,h.k,1,A,lda,B,ldb,0,C,h.n);
          if(v==3) b_mk(101,TA,TB,h.m,h.n,h.k,1,Ab,lda,Bb,ldb,0,C,h.n);
          if(v==4){tobf16(A,Ab,na);tobf16(B,Bb,nb);b_mk(101,TA,TB,h.m,h.n,h.k,1,Ab,lda,Bb,ldb,0,C,h.n);}
          if(v==5) b_ob(101,TA,TB,h.m,h.n,h.k,1,Ab,lda,Bb,ldb,0,C,h.n);
        }
        double dt=now()-t0; if(w) g[v]=fl*reps/dt/1e9; }
      tot[v]+=fl*reps/(fl*reps/ (g[v]>0?g[v]:1)/1) ; }
    printf("%-17s %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f\n",h.name,g[0],g[1],g[2],g[3],g[4],g[5]);
    free(A);free(B);free(C);free(Ab);free(Bb);}
  return 0;}
