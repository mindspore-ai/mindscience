/*
 * Copyright 2021 Gao's lab, Peking University, CCME. All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef COMMON_CUH
#define COMMON_CUH
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include "thrust/sort.h"
//存储各种常数
//圆周率
#define CONSTANT_Pi 3.1415926535897932f
//自然对数的底
#define CONSTANT_e 2.7182818284590452f
//玻尔兹曼常量（kcal.mol^-1.K^ -1）
//使用kcal为能量单位，因此kB=8.31441(J.mol^-1.K^-1)/4.18407(J/cal)/1000
#define CONSTANT_kB 0.00198716f
//程序中使用的单位时间与物理时间的换算1/20.455*dt=1 ps
#define CONSTANT_TIME_CONVERTION 20.455f
//程序中使用的单位压强与物理压强的换算
// 压强单位: bar -> kcal/mol/A^3
// (1 kcal/mol) * (4.184074e3 J/kcal) / (6.023e23 mol^-1) * (1e30 m^3/A^3) *
// (1e-5 bar/pa) 程序的压强/(kcal/mol/A^3 ) * CONSTANT_PRES_CONVERTION =
// 物理压强/bar
#define CONSTANT_PRES_CONVERTION 6.946827162543585e4f
// 物理压强/bar * CONSTANT_PRES_CONVERTION_INVERSE = 程序的压强/(kcal/mol/A^3 )
#define CONSTANT_PRES_CONVERTION_INVERSE 0.00001439506089041446f
//周期性盒子映射所使用的信息，最大的unsigned int
#define CONSTANT_UINT_MAX UINT_MAX
//周期性盒子映射所使用的信息，最大的unsigned int对应的float
#define CONSTANT_UINT_MAX_FLOAT 4294967296.0f
//周期性盒子映射所使用的信息，最大的unsigned int对应的倒数
#define CONSTANT_UINT_MAX_INVERSED 2.3283064365387e-10f

#define CHAR_LENGTH_MAX 256
struct CONSTANT {
  //数学常数
  const float pi = 3.1415926535897932f;
  const float e = 2.7182818284590452f;
  //物理常量
  const float kB =
      0.00198716f; //玻尔兹曼常量（kcal.mol^-1.K^ -1）
                   //使用kcal为能量单位，因此kB=8.31441(J.mol^-1.K^-1)/4.18407(J/cal)/1000
  const float time_convertion =
      20.455f; //程序中使用的单位时间与物理时间的换算1/20.455*dt=1 ps
  //周期性盒子映射所使用的信息
  const unsigned int uint_max = UINT_MAX;
  const float uint_max_float = 4294967296.0f;
  const float uint_max_inversed = (float)1. / 4294967296.;
};

//用于存储各种三维float矢量而定义的结构体
struct VECTOR {
  float x;
  float y;
  float z;
};
//与VECTOR结构体相关的重载运算符
//逐项相加
__device__ __host__ VECTOR operator+(const VECTOR &veca, const VECTOR &vecb);
__device__ __host__ VECTOR operator+(const VECTOR &veca, const float &b);
//点积
__device__ __host__ float operator*(const VECTOR &veca, const VECTOR &vecb);
//标量积
__device__ __host__ VECTOR operator*(const float &a, const VECTOR &vecb);
//逐项相除
__device__ __host__ VECTOR operator/(const VECTOR &veca, const VECTOR &vecb);
__device__ __host__ VECTOR operator/(const float &a, const VECTOR &vecb);
//逐项相减
__device__ __host__ VECTOR operator-(const VECTOR &veca, const VECTOR &vecb);
__device__ __host__ VECTOR operator-(const VECTOR &veca, const float &b);
//取负
__device__ __host__ VECTOR operator-(const VECTOR &vecb);
//外积
__device__ __host__ VECTOR operator^(const VECTOR &veca, const VECTOR &vecb);

//用于计算边界循环所定义的结构体
struct UNSIGNED_INT_VECTOR {
  unsigned int uint_x;
  unsigned int uint_y;
  unsigned int uint_z;
};

//用于计算边界循环或者一些三维数组大小所定义的结构体
struct INT_VECTOR {
  int int_x;
  int int_y;
  int int_z;
};

//用于计算周期性边界条件下的距离
//由整数坐标和转化系数求距离
__device__ __host__ VECTOR Get_Periodic_Displacement(
    const UNSIGNED_INT_VECTOR uvec_a, const UNSIGNED_INT_VECTOR uvec_b,
    const VECTOR scaler);
//由坐标和盒子长度求距离
__device__ __host__ VECTOR Get_Periodic_Displacement(const VECTOR vec_a,
                                                     const VECTOR vec_b,
                                                     const VECTOR box_length);
//由坐标,盒子长度,盒子长度的倒数求距离
__device__ __host__ VECTOR Get_Periodic_Displacement(
    const VECTOR vec_a, const VECTOR vec_b, const VECTOR box_length_inverse);

//使得某个向量的模不超过某个值，如果超过则方向不变地将模长放缩至该值
__device__ VECTOR Make_Vector_Not_Exceed_Value(VECTOR vector,
                                               const float value);

//用于记录原子组
struct ATOM_GROUP {
  int atom_numbers;
  int *atom_serial;
};

//用来重置一个已经分配过显存的列表：list。使用CUDA一维block和thread启用
void Reset_List(int *list, const int replace_element, const int element_numbers,
                const int threads = 1024);
__global__ void Reset_List(const int element_numbers, int *list,
                           const int replace_element);
void Reset_List(float *list, const float replace_element,
                const int element_numbers, const int threads = 1024);
__global__ void Reset_List(const int element_numbers, float *list,
                           const float replace_element);
//对一个列表的数值进行缩放
void Scale_List(float *list, const float scaler, const int element_numbers,
                const int threads = 1024);
__global__ void Scale_List(const int element_numbers, float *list,
                           float scaler);
//用来复制一个列表
__global__ void Copy_List(const int element_numbers, const int *origin_list,
                          int *list);
__global__ void Copy_List(const int element_numbers, const float *origin_list,
                          float *list);
//用来将一个列表中的每个元素取其倒数
__global__ void Inverse_List_Element(const int element_numbers,
                                     const float *origin_list, float *list);
//对一个列表求和，并将和记录在sum中
void Sum_Of_List(const int *list, int *sum, const int end, int threads = 1024);
void Sum_Of_List(const float *list, float *sum, const int end,
                 const int start = 0, int threads = 1024);
__global__ void Sum_Of_List(const int element_numbers, const int *list,
                            int *sum);
__global__ void Sum_Of_List(const int start, const int end, const float *list,
                            float *sum);
__global__ void Sum_Of_List(const int element_numbers, const float *list,
                            float *sum);
__global__ void Sum_Of_List(const int element_numbers, const VECTOR *list,
                            VECTOR *sum);

//用来将原子的真实坐标转换为unsigned
//int坐标,注意factor需要乘以0.5（保证越界坐标自然映回box）
__global__ void Crd_To_Uint_Crd(const int atom_numbers,
                                const VECTOR scale_factor, const VECTOR *crd,
                                UNSIGNED_INT_VECTOR *uint_crd);
//用来将坐标从真实坐标变为int坐标，factor不用乘以0.5，因为假设这类真实坐标总是比周期边界小得多。目前主要用于格林函数离散点的坐标映射
__global__ void Crd_To_Int_Crd(const int atom_numbers,
                               const VECTOR scale_factor, const VECTOR *crd,
                               INT_VECTOR *int_crd);
//用来对原子真实坐标进行周期性映射
__global__ void Crd_Periodic_Map(const int atom_numbers, VECTOR *crd,
                                 const VECTOR box_length);

//用来平移一组向量(CPU包装)
void Vector_Translation(const int vector_numbers, VECTOR *vec_list,
                        const VECTOR translation_vec, int threads_per_block);
//用来平移gpu上的一个平移向量（并非一组）（CPU包装）
void Vector_Translation(const int vector_numbers, VECTOR *vec_list,
                        const VECTOR *translation_vec, int threads_per_block);

//用来平移一组向量
__global__ void Vector_Translation(const int vector_numbers, VECTOR *vec_list,
                                   const VECTOR translation_vec);
// gpu上的一个平移向量（并非一组）
__global__ void Vector_Translation(const int vector_numbers, VECTOR *vec_list,
                                   const VECTOR *translation_vec);

//用于安全的显存和内存分配，以及打开文件
bool Malloc_Safely(void **address, size_t size);
bool Cuda_Malloc_Safely(void **address, size_t size);
bool Open_File_Safely(FILE **file, const char *file_name,
                      const char *open_type);

//用于生成高斯分布的随机数
//用seed初始化制定长度的随机数生成器，每个生成器一次可以生成按高斯分布的四个独立的数
__global__ void Setup_Rand_Normal_Kernel(const int float4_numbers,
                                         curandStatePhilox4_32_10_t *rand_state,
                                         const int seed);
//用生成器生成一次随机数，将其存入数组中
__global__ void Rand_Normal(const int float4_numbers,
                            curandStatePhilox4_32_10_t *rand_state,
                            float4 *rand_float4);

//用于GPU上的debug，将GPU上的信息打印出来
__global__ void Cuda_Debug_Print(float *x);
__global__ void Cuda_Debug_Print(VECTOR *x);
__global__ void Cuda_Debug_Print(int *x);

//用于做快速傅里叶变换前选择格点数目
int Get_Fft_Patameter(float length);
int Check_2357_Factor(int number);
#endif // COMMON_CUH(common.cuh)
