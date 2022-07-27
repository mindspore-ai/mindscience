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

#include "PME_force.cuh"

// constants
#define PI 3.1415926
#define INVSQRTPI 0.56418958835977
#define TWO_DIVIDED_BY_SQRT_PI 1.1283791670218446
__constant__ float PME_Ma[4] = {1.0 / 6.0, -0.5, 0.5, -1.0 / 6.0};
__constant__ float PME_Mb[4] = {0, 0.5, -1, 0.5};
__constant__ float PME_Mc[4] = {0, 0.5, 0, -0.5};
__constant__ float PME_Md[4] = {0, 1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0};
__constant__ float PME_dMa[4] = {0.5, -1.5, 1.5, -0.5};
__constant__ float PME_dMb[4] = {0, 1, -2, 1};
__constant__ float PME_dMc[4] = {0, 0.5, 0, -0.5};

// local functions
static float M_(float u, int n) {
  if (n == 2) {
    if (u > 2 || u < 0)
      return 0;
    return 1 - abs(u - 1);
  } else {
    return u / (n - 1) * M_(u, n - 1) + (n - u) / (n - 1) * M_(u - 1, n - 1);
  }
}

static cufftComplex expc(cufftComplex z) {
  cufftComplex res;
  float t = expf(z.x);
  sincosf(z.y, &res.y, &res.x);
  res.x *= t;
  res.y *= t;
  return res;
}

static float getb(int k, int NFFT, int B_order) {
  cufftComplex tempc, tempc2, res;
  float tempf;
  tempc2.x = 0;
  tempc2.y = 0;

  tempc.x = 0;
  tempc.y = 2 * (B_order - 1) * PI * k / NFFT;
  res = expc(tempc);

  for (int kk = 0; kk < (B_order - 1); kk++) {
    tempc.x = 0;
    tempc.y = 2 * PI * k / NFFT * kk;
    tempc = expc(tempc);
    tempf = M_(kk + 1, B_order);
    tempc2.x += tempf * tempc.x;
    tempc2.y += tempf * tempc.y;
  }
  res = cuCdivf(res, tempc2);
  return res.x * res.x + res.y * res.y;
}

static float Get_Beta(float cutoff, float tolerance) {
  float beta, low, high, tempf;
  int ilow, ihigh;

  high = 1.0;
  ihigh = 1;

  while (1) {
    tempf = erfc(high * cutoff) / cutoff;
    if (tempf <= tolerance)
      break;
    high *= 2;
    ihigh++;
  }

  ihigh += 50;
  low = 0.0;
  for (ilow = 1; ilow < ihigh; ilow++) {
    beta = (low + high) / 2;
    tempf = erfc(beta * cutoff) / cutoff;
    if (tempf >= tolerance)
      low = beta;
    else
      high = beta;
  }
  return beta;
}

static __global__ void device_add(float *ene, float factor, float *charge_sum) {
  ene[0] += factor * charge_sum[0] * charge_sum[0];
}

//////////////////////////////
void Particle_Mesh_Ewald::Initial(CONTROLLER *controller, int atom_numbers,
                                  VECTOR boxlength, float cutoff,
                                  const char *module_name) {
  if (module_name == NULL) {
    strcpy(this->module_name, "PME");
  } else {
    strcpy(this->module_name, module_name);
  }

  controller[0].printf("START INITIALIZING PME:\n");
  this->cutoff = cutoff;

  tolerance = 0.00001;
  if (controller[0].Command_Exist(this->module_name, "Direct_Tolerance"))
    tolerance =
        atof(controller[0].Command(this->module_name, "Direct_Tolerance"));

  fftx = -1;
  ffty = -1;
  fftz = -1;
  if (controller[0].Command_Exist(this->module_name, "fftx"))
    fftx = atoi(controller[0].Command(this->module_name, "fftx"));
  if (controller[0].Command_Exist(this->module_name, "ffty"))
    ffty = atoi(controller[0].Command(this->module_name, "ffty"));
  if (controller[0].Command_Exist(this->module_name, "fftz"))
    fftz = atoi(controller[0].Command(this->module_name, "fftz"));

  this->atom_numbers = atom_numbers;
  this->boxlength = boxlength;

  float volume = boxlength.x * boxlength.y * boxlength.z;

  if (fftx < 0)
    fftx = Get_Fft_Patameter(boxlength.x);

  if (ffty < 0)
    ffty = Get_Fft_Patameter(boxlength.y);

  if (fftz < 0)
    fftz = Get_Fft_Patameter(boxlength.z);

  controller[0].printf("    fftx: %d\n", fftx);
  controller[0].printf("    ffty: %d\n", ffty);
  controller[0].printf("    fftz: %d\n", fftz);

  PME_Nall = fftx * ffty * fftz;
  PME_Nin = ffty * fftz;
  PME_Nfft = fftx * ffty * (fftz / 2 + 1);
  PME_inverse_box_vector.x = (float)fftx / boxlength.x;
  PME_inverse_box_vector.y = (float)ffty / boxlength.y;
  PME_inverse_box_vector.z = (float)fftz / boxlength.z;

  beta = Get_Beta(cutoff, tolerance);
  controller[0].printf("    beta: %f\n", beta);

  neutralizing_factor = -0.5 * CONSTANT_Pi / (beta * beta * volume);
  Cuda_Malloc_Safely((void **)&charge_sum, sizeof(float));

  int i, kx, ky, kz, kxrp, kyrp, kzrp, index;
  cufftResult errP1, errP2;

  Cuda_Malloc_Safely((void **)&PME_uxyz,
                     sizeof(UNSIGNED_INT_VECTOR) * atom_numbers);
  Cuda_Malloc_Safely((void **)&PME_frxyz, sizeof(VECTOR) * atom_numbers);
  Reset_List<<<3 * atom_numbers / 32 + 1, 32>>>(3 * atom_numbers,
                                                (int *)PME_uxyz, 1 << 30);

  Cuda_Malloc_Safely((void **)&PME_Q, sizeof(float) * PME_Nall);
  Cuda_Malloc_Safely((void **)&PME_FQ, sizeof(cufftComplex) * PME_Nfft);
  Cuda_Malloc_Safely((void **)&PME_FBCFQ, sizeof(float) * PME_Nall);

  int **atom_near_cpu = NULL;
  Malloc_Safely((void **)&atom_near_cpu, sizeof(int *) * atom_numbers);
  Cuda_Malloc_Safely((void **)&PME_atom_near, sizeof(int *) * atom_numbers);
  for (i = 0; i < atom_numbers; i++) {
    Cuda_Malloc_Safely((void **)&atom_near_cpu[i], sizeof(int) * 64);
  }
  cudaMemcpy(PME_atom_near, atom_near_cpu, sizeof(int *) * atom_numbers,
             cudaMemcpyHostToDevice);
  free(atom_near_cpu);

  errP1 = cufftPlan3d(&PME_plan_r2c, fftx, ffty, fftz, CUFFT_R2C);
  errP2 = cufftPlan3d(&PME_plan_c2r, fftx, ffty, fftz, CUFFT_C2R);
  if (errP1 != CUFFT_SUCCESS || errP2 != CUFFT_SUCCESS) {
    controller[0].printf("    Error occurs when create fft plan of PME");
    getchar();
  }

  Cuda_Malloc_Safely((void **)&d_reciprocal_ene, sizeof(float));
  Cuda_Malloc_Safely((void **)&d_self_ene, sizeof(float));
  Cuda_Malloc_Safely((void **)&d_direct_ene, sizeof(float));
  Cuda_Malloc_Safely((void **)&d_direct_atom_energy,
                     sizeof(float) * atom_numbers);
  Cuda_Malloc_Safely((void **)&d_correction_atom_energy,
                     sizeof(float) * atom_numbers);
  Cuda_Malloc_Safely((void **)&d_correction_ene, sizeof(float));
  Cuda_Malloc_Safely((void **)&d_ee_ene, sizeof(float));

  UNSIGNED_INT_VECTOR *PME_kxyz_cpu = NULL;
  Cuda_Malloc_Safely((void **)&PME_kxyz, sizeof(UNSIGNED_INT_VECTOR) * 64);
  Malloc_Safely((void **)&PME_kxyz_cpu, sizeof(UNSIGNED_INT_VECTOR) * 64);

  for (kx = 0; kx < 4; kx++)
    for (ky = 0; ky < 4; ky++)
      for (kz = 0; kz < 4; kz++) {
        index = kx * 16 + ky * 4 + kz;
        PME_kxyz_cpu[index].uint_x = kx;
        PME_kxyz_cpu[index].uint_y = ky;
        PME_kxyz_cpu[index].uint_z = kz;
      }
  cudaMemcpy(PME_kxyz, PME_kxyz_cpu, sizeof(UNSIGNED_INT_VECTOR) * 64,
             cudaMemcpyHostToDevice);
  free(PME_kxyz_cpu);

  float *B1 = NULL, *B2 = NULL, *B3 = NULL, *h_PME_BC = NULL, *h_PME_BC0 = NULL;
  B1 = (float *)malloc(sizeof(float) * fftx);
  B2 = (float *)malloc(sizeof(float) * ffty);
  B3 = (float *)malloc(sizeof(float) * fftz);
  h_PME_BC0 = (float *)malloc(sizeof(float) * PME_Nfft);
  h_PME_BC = (float *)malloc(sizeof(float) * PME_Nfft);
  if (B1 == NULL || B2 == NULL || B3 == NULL || h_PME_BC0 == NULL ||
      h_PME_BC == NULL) {
    controller[0].printf("    Error occurs when malloc PME_BC of PME");
    getchar();
  }
  for (kx = 0; kx < fftx; kx++) {
    B1[kx] = getb(kx, fftx, 4);
  }

  for (ky = 0; ky < ffty; ky++) {
    B2[ky] = getb(ky, ffty, 4);
  }

  for (kz = 0; kz < fftz; kz++) {
    B3[kz] = getb(kz, fftz, 4);
  }

  float mprefactor = PI * PI / -beta / beta;
  float msq;
  for (kx = 0; kx < fftx; kx++) {
    kxrp = kx;
    if (kx > fftx / 2)
      kxrp = fftx - kx;
    for (ky = 0; ky < ffty; ky++) {
      kyrp = ky;
      if (ky > ffty / 2)
        kyrp = ffty - ky;
      for (kz = 0; kz <= fftz / 2; kz++) {
        kzrp = kz;

        msq = kxrp * kxrp / boxlength.x / boxlength.x +
              kyrp * kyrp / boxlength.y / boxlength.y +
              kzrp * kzrp / boxlength.z / boxlength.z;

        index = kx * ffty * (fftz / 2 + 1) + ky * (fftz / 2 + 1) + kz;

        if (kx + ky + kz == 0)
          h_PME_BC[index] = 0;
        else
          h_PME_BC[index] =
              (float)1.0 / PI / msq * exp(mprefactor * msq) / volume;

        h_PME_BC0[index] = B1[kx] * B2[ky] * B3[kz];
        h_PME_BC[index] *= h_PME_BC0[index];
      }
    }
  }

  Cuda_Malloc_Safely((void **)&PME_BC, sizeof(float) * PME_Nfft);
  Cuda_Malloc_Safely((void **)&PME_BC0, sizeof(float) * PME_Nfft);
  cudaMemcpy(PME_BC, h_PME_BC, sizeof(float) * PME_Nfft,
             cudaMemcpyHostToDevice);
  cudaMemcpy(PME_BC0, h_PME_BC0, sizeof(float) * PME_Nfft,
             cudaMemcpyHostToDevice);
  free(B1);
  free(B2);
  free(B3);
  free(h_PME_BC0);
  free(h_PME_BC);

  is_initialized = 1;
  if (is_initialized && !is_controller_printf_initialized) {
    controller[0].Step_Print_Initial(this->module_name, "%.2f");
    is_controller_printf_initialized = 1;
    controller[0].printf("    structure last modify date is %d\n",
                         last_modify_date);
  }
  controller[0].printf("END INITIALIZING PME\n\n");
}

void Particle_Mesh_Ewald::Clear() {
  if (is_initialized) {
    is_initialized = 0;
    cudaFree(PME_uxyz);
    cudaFree(PME_kxyz);
    cudaFree(PME_frxyz);
    cudaFree(PME_Q);
    cudaFree(PME_FQ);
    cudaFree(PME_FBCFQ);
    cudaFree(PME_BC);
    cudaFree(PME_BC0);
    cudaFree(charge_sum);

    PME_uxyz = NULL;
    PME_kxyz = NULL;
    PME_frxyz = NULL;
    PME_Q = NULL;
    PME_FQ = NULL;
    PME_FBCFQ = NULL;
    PME_BC = NULL;
    PME_BC0 = NULL;
    charge_sum = NULL;

    int **atom_near_cpu = NULL;
    Malloc_Safely((void **)&atom_near_cpu, sizeof(int *) * atom_numbers);
    cudaMemcpy(atom_near_cpu, PME_atom_near, sizeof(int *) * atom_numbers,
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < atom_numbers; i++) {
      cudaFree(atom_near_cpu[i]);
    }
    cudaFree(PME_atom_near);
    PME_atom_near = NULL;
    free(atom_near_cpu);

    cufftDestroy(PME_plan_r2c);
    cufftDestroy(PME_plan_c2r);

    cudaFree(d_reciprocal_ene);
    cudaFree(d_self_ene);
    cudaFree(d_direct_ene);
    cudaFree(d_direct_atom_energy);
    cudaFree(d_correction_atom_energy);
    cudaFree(d_correction_ene);
    cudaFree(d_ee_ene);

    d_reciprocal_ene = NULL;
    d_self_ene = NULL;
    d_direct_ene = NULL;
    d_direct_atom_energy = NULL;
    d_correction_atom_energy = NULL;
    d_correction_ene = NULL;
    d_ee_ene = NULL;
  }
}

__global__ void
PME_Atom_Near(const UNSIGNED_INT_VECTOR *uint_crd, int **PME_atom_near,
              const int PME_Nin, const float periodic_factor_inverse_x,
              const float periodic_factor_inverse_y,
              const float periodic_factor_inverse_z, const int atom_numbers,
              const int fftx, const int ffty, const int fftz,
              const UNSIGNED_INT_VECTOR *PME_kxyz,
              UNSIGNED_INT_VECTOR *PME_uxyz, VECTOR *PME_frxyz) {
  int atom = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom < atom_numbers) {
    UNSIGNED_INT_VECTOR *temp_uxyz = &PME_uxyz[atom];
    int k, tempux, tempuy, tempuz;
    float tempf;
    tempf = (float)uint_crd[atom].uint_x * periodic_factor_inverse_x;
    tempux = (int)tempf;
    PME_frxyz[atom].x = tempf - tempux;

    tempf = (float)uint_crd[atom].uint_y * periodic_factor_inverse_y;
    tempuy = (int)tempf;
    PME_frxyz[atom].y = tempf - tempuy;

    tempf = (float)uint_crd[atom].uint_z * periodic_factor_inverse_z;
    tempuz = (int)tempf;
    PME_frxyz[atom].z = tempf - tempuz;

    if (tempux != (*temp_uxyz).uint_x || tempuy != (*temp_uxyz).uint_y ||
        tempuz != (*temp_uxyz).uint_z) {
      (*temp_uxyz).uint_x = tempux;
      (*temp_uxyz).uint_y = tempuy;
      (*temp_uxyz).uint_z = tempuz;
      int *temp_near = PME_atom_near[atom];
      int kx, ky, kz;
      for (k = 0; k < 64; k++) {
        UNSIGNED_INT_VECTOR temp_kxyz = PME_kxyz[k];

        kx = tempux - temp_kxyz.uint_x;

        if (kx < 0)
          kx += fftx;
        if (kx >= fftx)
          kx -= fftx;
        ky = tempuy - temp_kxyz.uint_y;
        if (ky < 0)
          ky += ffty;
        if (ky >= ffty)
          ky -= ffty;
        kz = tempuz - temp_kxyz.uint_z;
        if (kz < 0)
          kz += fftz;
        if (kz >= fftz)
          kz -= fftz;
        temp_near[k] = kx * PME_Nin + ky * fftz + kz;
      }
    }
  }
}

__global__ void PME_Q_Spread(int **PME_atom_near, const float *charge,
                             const VECTOR *PME_frxyz, float *PME_Q,
                             const UNSIGNED_INT_VECTOR *PME_kxyz,
                             const int atom_numbers) {
  int atom = blockDim.x * blockIdx.x + threadIdx.x;

  if (atom < atom_numbers) {
    int k;
    float tempf, tempQ, tempf2;
    int *temp_near = PME_atom_near[atom];
    VECTOR temp_frxyz = PME_frxyz[atom];
    float tempcharge = charge[atom];

    UNSIGNED_INT_VECTOR temp_kxyz;
    unsigned int kx;

    for (k = threadIdx.y; k < 64; k = k + blockDim.y) {
      temp_kxyz = PME_kxyz[k];
      kx = temp_kxyz.uint_x;
      tempf = (temp_frxyz.x);
      tempf2 = tempf * tempf;
      tempf = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 +
              PME_Mc[kx] * tempf + PME_Md[kx];

      tempQ = tempcharge * tempf;

      kx = temp_kxyz.uint_y;
      tempf = (temp_frxyz.y);
      tempf2 = tempf * tempf;
      tempf = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 +
              PME_Mc[kx] * tempf + PME_Md[kx];

      tempQ = tempQ * tempf;

      kx = temp_kxyz.uint_z;
      tempf = (temp_frxyz.z);
      tempf2 = tempf * tempf;
      tempf = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 +
              PME_Mc[kx] * tempf + PME_Md[kx];
      tempQ = tempQ * tempf;

      atomicAdd(&PME_Q[temp_near[k]], tempQ);
    }
  }
}

__global__ void PME_BCFQ(cufftComplex *PME_FQ, float *PME_BC, int PME_Nfft) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < PME_Nfft) {
    float tempf = PME_BC[index];
    cufftComplex tempc = PME_FQ[index];
    PME_FQ[index].x = tempc.x * tempf;
    PME_FQ[index].y = tempc.y * tempf;
  }
}

static __global__ void PME_Final(int **PME_atom_near, const float *charge,
                                 const float *PME_Q, VECTOR *force,
                                 const VECTOR *PME_frxyz,
                                 const UNSIGNED_INT_VECTOR *PME_kxyz,
                                 const VECTOR PME_inverse_box_vector,
                                 const int atom_numbers) {
  int atom = blockDim.y * blockIdx.y + threadIdx.y;
  if (atom < atom_numbers) {
    int k, kx;
    float tempdQx, tempdQy, tempdQz, tempdx, tempdy, tempdz, tempx, tempy,
        tempz, tempdQf;
    float tempf, tempf2;
    float tempnvdx = 0.0f;
    float tempnvdy = 0.0f;
    float tempnvdz = 0.0f;
    float temp_charge = charge[atom];
    int *temp_near = PME_atom_near[atom];
    UNSIGNED_INT_VECTOR temp_kxyz;
    VECTOR temp_frxyz = PME_frxyz[atom];
    for (k = threadIdx.x; k < 64; k = k + blockDim.x) {
      temp_kxyz = PME_kxyz[k];
      tempdQf = -PME_Q[temp_near[k]] * temp_charge;

      kx = temp_kxyz.uint_x;
      tempf = (temp_frxyz.x);
      tempf2 = tempf * tempf;
      tempx = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 +
              PME_Mc[kx] * tempf + PME_Md[kx];
      tempdx = PME_dMa[kx] * tempf2 + PME_dMb[kx] * tempf + PME_dMc[kx];

      kx = temp_kxyz.uint_y;
      tempf = (temp_frxyz.y);
      tempf2 = tempf * tempf;
      tempy = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 +
              PME_Mc[kx] * tempf + PME_Md[kx];
      tempdy = PME_dMa[kx] * tempf2 + PME_dMb[kx] * tempf + PME_dMc[kx];

      kx = temp_kxyz.uint_z;
      tempf = (temp_frxyz.z);
      tempf2 = tempf * tempf;
      tempz = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 +
              PME_Mc[kx] * tempf + PME_Md[kx];
      tempdz = PME_dMa[kx] * tempf2 + PME_dMb[kx] * tempf + PME_dMc[kx];

      tempdQx = tempdx * tempy * tempz * PME_inverse_box_vector.x;
      tempdQy = tempdy * tempx * tempz * PME_inverse_box_vector.y;
      tempdQz = tempdz * tempx * tempy * PME_inverse_box_vector.z;

      tempnvdx += tempdQf * tempdQx;
      tempnvdy += tempdQf * tempdQy;
      tempnvdz += tempdQf * tempdQz;
    }
    for (int offset = 4; offset > 0; offset /= 2) {
      tempnvdx += __shfl_xor_sync(0xFFFFFFFF, tempnvdx, offset, 8);
      tempnvdy += __shfl_xor_sync(0xFFFFFFFF, tempnvdy, offset, 8);
      tempnvdz += __shfl_xor_sync(0xFFFFFFFF, tempnvdz, offset, 8);
    }

    if (threadIdx.x == 0) {
      force[atom].x = force[atom].x + tempnvdx;
      force[atom].y = force[atom].y + tempnvdy;
      force[atom].z = force[atom].z + tempnvdz;
    }
  }
}

__global__ void PME_Energy_Product(const int element_number, const float *list1,
                                   const float *list2, float *sum) {
  if (threadIdx.x == 0) {
    sum[0] = 0.;
  }
  __syncthreads();
  float lin = 0.0;
  for (int i = threadIdx.x; i < element_number; i = i + blockDim.x) {
    lin = lin + list1[i] * list2[i];
  }
  atomicAdd(sum, lin);
}

static __global__ void
PME_Direct_Atom_Energy(const int atom_numbers, const ATOM_GROUP *nl,
                       const UNSIGNED_INT_VECTOR *uint_crd,
                       const VECTOR boxlength, const float *charge,
                       const float beta, const float cutoff_square,
                       float *direct_ene) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    ATOM_GROUP nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UNSIGNED_INT_VECTOR r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    float dr2;
    float dr_abs;
    float ene_temp;
    float charge_i = charge[atom_i];
    float ene_lin = 0.;

    for (int j = threadIdx.y; j < N; j = j + blockDim.y) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength.x * int_x;
      dr.y = boxlength.y * int_y;
      dr.z = boxlength.z * int_z;

      dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
      if (dr2 < cutoff_square) {
        dr_abs = norm3df(dr.x, dr.y, dr.z);
        ene_temp = charge_i * charge[atom_j] * erfcf(beta * dr_abs) / dr_abs;
        ene_lin = ene_lin + ene_temp;
      }
    } // atom_j cycle
    atomicAdd(&direct_ene[atom_i], ene_lin);
  }
}

static __global__ void PME_Excluded_Force_With_Atom_Energy_Correction(
    const int atom_numbers, const UNSIGNED_INT_VECTOR *uint_crd,
    const VECTOR sacler, const float *charge, const float pme_beta,
    const float sqrt_pi, const int *excluded_list_start,
    const int *excluded_list, const int *excluded_atom_numbers, VECTOR *frc,
    float *ene) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    int excluded_numbers = excluded_atom_numbers[atom_i];
    if (excluded_numbers > 0) {
      int list_start = excluded_list_start[atom_i];
      int list_end = list_start + excluded_numbers;
      int atom_j;
      int int_x;
      int int_y;
      int int_z;

      float charge_i = charge[atom_i];
      float charge_j;
      float dr_abs;
      float beta_dr;

      UNSIGNED_INT_VECTOR r1 = uint_crd[atom_i], r2;
      VECTOR dr;
      float dr2;

      float frc_abs = 0.;
      VECTOR frc_lin;
      VECTOR frc_record = {0., 0., 0.};
      float ene_lin = 0.;

      for (int i = list_start; i < list_end; i = i + 1) {
        atom_j = excluded_list[i];
        r2 = uint_crd[atom_j];
        charge_j = charge[atom_j];

        int_x = r2.uint_x - r1.uint_x;
        int_y = r2.uint_y - r1.uint_y;
        int_z = r2.uint_z - r1.uint_z;
        dr.x = sacler.x * int_x;
        dr.y = sacler.y * int_y;
        dr.z = sacler.z * int_z;
        dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
        //假设剔除表中的原子对距离总是小于cutoff的，正常体系

        dr_abs = sqrtf(dr2);
        beta_dr = pme_beta * dr_abs;
        // sqrt_pi= 2/sqrt(3.141592654);
        frc_abs = beta_dr * sqrt_pi * expf(-beta_dr * beta_dr) + erfcf(beta_dr);
        frc_abs = (frc_abs - 1.) / dr2 / dr_abs;
        frc_abs = -charge_i * charge_j * frc_abs;
        frc_lin.x = frc_abs * dr.x;
        frc_lin.y = frc_abs * dr.y;
        frc_lin.z = frc_abs * dr.z;
        ene_lin -= charge_i * charge_j * erff(beta_dr) / dr_abs;

        frc_record.x = frc_record.x + frc_lin.x;
        frc_record.y = frc_record.y + frc_lin.y;
        frc_record.z = frc_record.z + frc_lin.z;

        atomicAdd(&frc[atom_j].x, -frc_lin.x);
        atomicAdd(&frc[atom_j].y, -frc_lin.y);
        atomicAdd(&frc[atom_j].z, -frc_lin.z);
      } // atom_j cycle
      atomicAdd(&frc[atom_i].x, frc_record.x);
      atomicAdd(&frc[atom_i].y, frc_record.y);
      atomicAdd(&frc[atom_i].z, frc_record.z);
      atomicAdd(ene + atom_i, ene_lin);
    } // if need excluded
  }
}

void Particle_Mesh_Ewald::PME_Excluded_Force_With_Atom_Energy(
    const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR sacler,
    const float *charge, const int *excluded_list_start,
    const int *excluded_list, const int *excluded_atom_numbers, VECTOR *frc,
    float *atom_energy) {
  if (is_initialized) {
    Reset_List<<<ceilf((float)atom_numbers / 1024.0f), 1024>>>(
        atom_numbers, d_correction_atom_energy, 0.0f);
    PME_Excluded_Force_With_Atom_Energy_Correction<<<
        ceilf((float)atom_numbers / 128), 128>>>(
        atom_numbers, uint_crd, sacler, charge, beta, TWO_DIVIDED_BY_SQRT_PI,
        excluded_list_start, excluded_list, excluded_atom_numbers, frc,
        atom_energy);
  }
}

static __global__ void PME_Add_Energy_To_Virial(float *d_virial,
                                                float *d_direct_ene,
                                                float *d_correction_ene,
                                                float *d_self_ene,
                                                float *d_reciprocal_ene) {
  d_virial[0] += d_direct_ene[0] + d_correction_ene[0] + d_self_ene[0] +
                 d_reciprocal_ene[0];
}

static __global__ void PME_Add_Energy_To_Potential(float *d_virial,
                                                   float *d_correction_ene,
                                                   float *d_self_ene,
                                                   float *d_reciprocal_ene) {
  d_virial[0] += d_correction_ene[0] + d_self_ene[0] + d_reciprocal_ene[0];
}

void Particle_Mesh_Ewald::PME_Reciprocal_Force_With_Energy_And_Virial(
    const UNSIGNED_INT_VECTOR *uint_crd, const float *charge, VECTOR *force,
    int need_virial, int need_energy, float *d_virial, float *d_potential) {
  if (is_initialized) {
    PME_Atom_Near<<<atom_numbers / 32 + 1, 32>>>(
        uint_crd, PME_atom_near, PME_Nin, CONSTANT_UINT_MAX_INVERSED * fftx,
        CONSTANT_UINT_MAX_INVERSED * ffty, CONSTANT_UINT_MAX_INVERSED * fftz,
        atom_numbers, fftx, ffty, fftz, PME_kxyz, PME_uxyz, PME_frxyz);

    Reset_List<<<PME_Nall / 1024 + 1, 1024>>>(PME_Nall, PME_Q, 0);

    PME_Q_Spread<<<atom_numbers / thread_PME.x + 1, thread_PME>>>(
        PME_atom_near, charge, PME_frxyz, PME_Q, PME_kxyz, atom_numbers);

    cufftExecR2C(PME_plan_r2c, (float *)PME_Q, (cufftComplex *)PME_FQ);

    PME_BCFQ<<<PME_Nfft / 1024 + 1, 1024>>>(PME_FQ, PME_BC, PME_Nfft);

    cufftExecC2R(PME_plan_c2r, (cufftComplex *)PME_FQ, (float *)PME_FBCFQ);

    PME_Final<<<{1, atom_numbers / thread_PME.x + 1}, thread_PME>>>(
        PME_atom_near, charge, PME_FBCFQ, force, PME_frxyz, PME_kxyz,
        PME_inverse_box_vector, atom_numbers);

    if (need_virial > 0 || need_energy > 0) {
      PME_Energy_Product<<<1, 1024>>>(PME_Nall, PME_Q, PME_FBCFQ,
                                      d_reciprocal_ene);
      Scale_List<<<1, 1>>>(1, d_reciprocal_ene, 0.5);

      PME_Energy_Product<<<1, 1024>>>(atom_numbers, charge, charge, d_self_ene);
      Scale_List<<<1, 1>>>(1, d_self_ene, -beta / sqrtf(PI));

      Sum_Of_List<<<1, 1024>>>(atom_numbers, charge, charge_sum);
      device_add<<<1, 1>>>(d_self_ene, neutralizing_factor, charge_sum);

      Sum_Of_List<<<1, 1024>>>(atom_numbers, d_direct_atom_energy,
                               d_direct_ene);
      Sum_Of_List<<<1, 1024>>>(atom_numbers, d_correction_atom_energy,
                               d_correction_ene);

      if (need_energy > 0)
        PME_Add_Energy_To_Potential<<<1, 1>>>(d_potential, d_correction_ene,
                                              d_self_ene, d_reciprocal_ene);
      if (need_virial > 0)
        PME_Add_Energy_To_Virial<<<1, 1>>>(d_virial, d_direct_ene,
                                           d_correction_ene, d_self_ene,
                                           d_reciprocal_ene);
    }
  }
}

static __global__ void PME_Excluded_Energy_Correction(
    const int atom_numbers, const UNSIGNED_INT_VECTOR *uint_crd,
    const VECTOR sacler, const float *charge, const float pme_beta,
    const float sqrt_pi, const int *excluded_list_start,
    const int *excluded_list, const int *excluded_atom_numbers, float *ene) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    int excluded_number = excluded_atom_numbers[atom_i];
    if (excluded_number > 0) {
      int list_start = excluded_list_start[atom_i];
      int list_end = list_start + excluded_number;
      int atom_j;
      int int_x;
      int int_y;
      int int_z;

      float charge_i = charge[atom_i];
      float charge_j;
      float dr_abs;
      float beta_dr;

      UNSIGNED_INT_VECTOR r1 = uint_crd[atom_i], r2;
      VECTOR dr;
      float dr2;

      float ene_lin = 0.;

      for (int i = list_start; i < list_end; i = i + 1) {
        atom_j = excluded_list[i];
        r2 = uint_crd[atom_j];
        charge_j = charge[atom_j];

        int_x = r2.uint_x - r1.uint_x;
        int_y = r2.uint_y - r1.uint_y;
        int_z = r2.uint_z - r1.uint_z;
        dr.x = sacler.x * int_x;
        dr.y = sacler.y * int_y;
        dr.z = sacler.z * int_z;
        dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
        //假设剔除表中的原子对距离总是小于cutoff的，正常体系
        dr_abs = sqrtf(dr2);
        beta_dr = pme_beta * dr_abs;

        ene_lin -= charge_i * charge_j * erff(beta_dr) / dr_abs;
      } // atom_j cycle
      atomicAdd(ene + atom_i, ene_lin);
    } // if need excluded
  }
}

/*float Particle_Mesh_Ewald::Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd,
const float *charge, const ATOM_GROUP *nl, const VECTOR scaler, const int
*excluded_list_start, const int *excluded_list, const int
*excluded_atom_numbers, int is_download)
{
        PME_Atom_Near << <atom_numbers / 32 + 1, 32 >> >
                (uint_crd, PME_atom_near, PME_Nin,
                CONSTANT_UINT_MAX_INVERSED * fftx, CONSTANT_UINT_MAX_INVERSED *
ffty, CONSTANT_UINT_MAX_INVERSED * fftz, atom_numbers, fftx, ffty, fftz,
                PME_kxyz, PME_uxyz, PME_frxyz);

        Reset_List << < PME_Nall / 1024 + 1, 1024 >> >(PME_Nall, PME_Q, 0);

        PME_Q_Spread << < atom_numbers / thread_PME.x + 1, thread_PME >> >
                (PME_atom_near, charge, PME_frxyz,
                PME_Q, PME_kxyz, atom_numbers);

        cufftExecR2C(PME_plan_r2c, (float*)PME_Q, (cufftComplex*)PME_FQ);


        PME_BCFQ << < PME_Nfft / 1024 + 1, 1024 >> > (PME_FQ, PME_BC, PME_Nfft);

        cufftExecC2R(PME_plan_c2r, (cufftComplex*)PME_FQ, (float*)PME_FBCFQ);

        PME_Energy_Product << < 1, 1024 >> >(PME_Nall, PME_Q, PME_FBCFQ,
d_reciprocal_ene); Scale_List << <1, 1 >> >(1, d_reciprocal_ene, 0.5);

        PME_Energy_Product << < 1, 1024 >> >(atom_numbers, charge, charge,
d_self_ene); Scale_List << <1, 1 >> >(1, d_self_ene, -beta / sqrtf(PI));

        Sum_Of_List << <1, 1024 >> >(atom_numbers, charge, charge_sum);
        device_add << <1, 1 >> >(d_self_ene, neutralizing_factor, charge_sum);

        Reset_List << <ceilf((float)atom_numbers / 1024.0f), 1024 >>
>(atom_numbers, d_direct_atom_energy, 0.0f); PME_Direct_Atom_Energy << <
atom_numbers / thread_PME.x + 1, thread_PME >> > (atom_numbers, nl, uint_crd,
scaler, charge, beta, cutoff*cutoff, d_direct_atom_energy); Sum_Of_List << <1,
1024 >> >(atom_numbers, d_direct_atom_energy, d_direct_ene);

        Reset_List << <ceilf((float)atom_numbers / 1024.0f), 1024 >>
>(atom_numbers, d_correction_atom_energy, 0.0f); PME_Excluded_Energy_Correction
<< < atom_numbers / 32 + 1, 32 >> > (atom_numbers, uint_crd, scaler, charge,
beta, sqrtf(PI), excluded_list_start, excluded_list, excluded_atom_numbers,
d_correction_atom_energy); Sum_Of_List << <1, 1024 >> >(atom_numbers,
d_correction_atom_energy, d_correction_ene);

        if (is_download)
        {
                cudaMemcpy(&reciprocal_ene, d_reciprocal_ene, sizeof(float),
cudaMemcpyDeviceToHost); cudaMemcpy(&self_ene, d_self_ene, sizeof(float),
cudaMemcpyDeviceToHost); cudaMemcpy(&direct_ene, d_direct_ene, sizeof(float),
cudaMemcpyDeviceToHost); cudaMemcpy(&correction_ene, d_correction_ene,
sizeof(float), cudaMemcpyDeviceToHost); ee_ene = reciprocal_ene + self_ene +
direct_ene + correction_ene; return ee_ene;
        }
        else
        {
                return 0;
        }
}*/

float Particle_Mesh_Ewald::Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd,
                                      const float *charge, const ATOM_GROUP *nl,
                                      const VECTOR scaler,
                                      const int *excluded_list_start,
                                      const int *excluded_list,
                                      const int *excluded_atom_numbers,
                                      int which_part, int is_download) {
  if (is_initialized) {
    if (which_part < 0 || which_part > 4) {
      printf("Error: PME Energy part %d is not allowed.\n", which_part);
      getchar();
      exit(0);
    }
    if (which_part == TOTAL || which_part == RECIPROCAL) {
      PME_Atom_Near<<<atom_numbers / 32 + 1, 32>>>(
          uint_crd, PME_atom_near, PME_Nin, CONSTANT_UINT_MAX_INVERSED * fftx,
          CONSTANT_UINT_MAX_INVERSED * ffty, CONSTANT_UINT_MAX_INVERSED * fftz,
          atom_numbers, fftx, ffty, fftz, PME_kxyz, PME_uxyz, PME_frxyz);

      Reset_List<<<PME_Nall / 1024 + 1, 1024>>>(PME_Nall, PME_Q, 0);

      PME_Q_Spread<<<atom_numbers / thread_PME.x + 1, thread_PME>>>(
          PME_atom_near, charge, PME_frxyz, PME_Q, PME_kxyz, atom_numbers);

      cufftExecR2C(PME_plan_r2c, (float *)PME_Q, (cufftComplex *)PME_FQ);

      PME_BCFQ<<<PME_Nfft / 1024 + 1, 1024>>>(PME_FQ, PME_BC, PME_Nfft);

      cufftExecC2R(PME_plan_c2r, (cufftComplex *)PME_FQ, (float *)PME_FBCFQ);

      PME_Energy_Product<<<1, 1024>>>(PME_Nall, PME_Q, PME_FBCFQ,
                                      d_reciprocal_ene);
      Scale_List<<<1, 1>>>(1, d_reciprocal_ene, 0.5);
    }

    if (which_part == TOTAL || which_part == SELF) {
      PME_Energy_Product<<<1, 1024>>>(atom_numbers, charge, charge, d_self_ene);
      Scale_List<<<1, 1>>>(1, d_self_ene, -beta / sqrtf(PI));

      Sum_Of_List<<<1, 1024>>>(atom_numbers, charge, charge_sum);
      device_add<<<1, 1>>>(d_self_ene, neutralizing_factor, charge_sum);
    }

    if (which_part == TOTAL || which_part == DIRECT) {
      Reset_List<<<ceilf((float)atom_numbers / 1024.0f), 1024>>>(
          atom_numbers, d_direct_atom_energy, 0.0f);
      PME_Direct_Atom_Energy<<<atom_numbers / thread_PME.x + 1, thread_PME>>>(
          atom_numbers, nl, uint_crd, scaler, charge, beta, cutoff * cutoff,
          d_direct_atom_energy);
      Sum_Of_List<<<1, 1024>>>(atom_numbers, d_direct_atom_energy,
                               d_direct_ene);
    }

    if (which_part == TOTAL || which_part == CORRECTION) {
      Reset_List<<<ceilf((float)atom_numbers / 1024.0f), 1024>>>(
          atom_numbers, d_correction_atom_energy, 0.0f);
      PME_Excluded_Energy_Correction<<<atom_numbers / 32 + 1, 32>>>(
          atom_numbers, uint_crd, scaler, charge, beta, sqrtf(PI),
          excluded_list_start, excluded_list, excluded_atom_numbers,
          d_correction_atom_energy);
      Sum_Of_List<<<1, 1024>>>(atom_numbers, d_correction_atom_energy,
                               d_correction_ene);
    }

    if (is_download) {
      if (which_part == TOTAL || which_part == RECIPROCAL)
        cudaMemcpy(&reciprocal_ene, d_reciprocal_ene, sizeof(float),
                   cudaMemcpyDeviceToHost);

      if (which_part == TOTAL || which_part == SELF)
        cudaMemcpy(&self_ene, d_self_ene, sizeof(float),
                   cudaMemcpyDeviceToHost);

      if (which_part == TOTAL || which_part == DIRECT)
        cudaMemcpy(&direct_ene, d_direct_ene, sizeof(float),
                   cudaMemcpyDeviceToHost);

      if (which_part == TOTAL || which_part == CORRECTION)
        cudaMemcpy(&correction_ene, d_correction_ene, sizeof(float),
                   cudaMemcpyDeviceToHost);

      if (which_part == TOTAL) {
        ee_ene = reciprocal_ene + self_ene + direct_ene + correction_ene;
        return ee_ene;
      } else if (which_part == RECIPROCAL) {
        return reciprocal_ene;
      } else if (which_part == SELF) {
        return self_ene;
      } else if (which_part == DIRECT) {
        return direct_ene;
      } else {
        return correction_ene;
      }
    } else {
      return 0;
    }
  } else {
    return NAN;
  }
}

void Particle_Mesh_Ewald::Update_Volume(VECTOR box_length) {
  Update_Box_Length(boxlength);
}

__global__ void up_box_bc(int fftx, int ffty, int fftz, float *PME_BC,
                          float *PME_BC0, float mprefactor, VECTOR boxlength,
                          float volume) {
  int kx, ky, kz, kxrp, kyrp, kzrp, index;
  float msq;
  for (kx = blockIdx.x * blockDim.x + threadIdx.x; kx < fftx;
       kx += blockDim.x * gridDim.x) {
    kxrp = kx;
    if (kx > fftx / 2)
      kxrp = fftx - kx;
    for (ky = blockIdx.y * blockDim.y + threadIdx.y; ky < ffty;
         ky += blockDim.y * gridDim.y) {
      kyrp = ky;
      if (ky > fftx / 2)
        kyrp = ffty - ky;
      for (kz = threadIdx.z; kz <= fftz / 2; kz += blockDim.z) {
        kzrp = kz;
        msq = kxrp * kxrp / boxlength.x / boxlength.x +
              kyrp * kyrp / boxlength.y / boxlength.y +
              kzrp * kzrp / boxlength.z / boxlength.z;

        index = kx * ffty * (fftz / 2 + 1) + ky * (fftz / 2 + 1) + kz;

        if (kx + ky + kz == 0)
          PME_BC[index] = 0;
        else
          PME_BC[index] = (float)1.0 / PI / msq * exp(mprefactor * msq) /
                          volume * PME_BC0[index];
      }
    }
  }
}

void Particle_Mesh_Ewald::Update_Box_Length(VECTOR boxlength) {
  float volume = boxlength.x * boxlength.y * boxlength.z;
  PME_inverse_box_vector.x = (float)fftx / boxlength.x;
  PME_inverse_box_vector.y = (float)ffty / boxlength.y;
  PME_inverse_box_vector.z = (float)fftz / boxlength.z;
  neutralizing_factor = -0.5 * CONSTANT_Pi / (beta * beta * volume);
  float mprefactor = PI * PI / -beta / beta;
  up_box_bc<<<{20, 20}, {8, 8, 16}>>>(fftx, ffty, fftz, PME_BC, PME_BC0,
                                      mprefactor, boxlength, volume);
}
