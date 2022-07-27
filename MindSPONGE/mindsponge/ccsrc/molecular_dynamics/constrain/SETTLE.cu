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

#include "SETTLE.cuh"

//对几何信息进行转化
//输入：rAB、rAC、rBC：三角形三边长
//输入：mA、mB、mC：ABC三个的质量
//输出：ra rb rc rd re：位置参数，当刚体三角形质心放置于原点时
// A点放置于(0,ra,0)，B点放置于(rc, rb, 0)，C点放置于(rd, re, 0)
__device__ __host__ void Get_Rabcde_From_SSS(float rAB, float rAC, float rBC,
                                             float mA, float mB, float mC,
                                             float &ra, float &rb, float &rc,
                                             float &rd, float &re) {
  float mTotal = mA + mB + mC;
  float Ax = 0;
  float Ay = 0;
  float Bx = -rAB;
  float By = 0;
  float costemp = (rBC * rBC - rAC * rAC - rAB * rAB) / (2 * rAC * rAB);
  float Cx = rAC * costemp;
  float sintemp = sqrtf(1.0f - costemp * costemp);
  float Cy = rAC * sintemp;

  float Ox = (Bx * mB + Cx * mC) / mTotal;
  float Oy = Cy * mC / mTotal;

  Ax -= Ox;
  Ay -= Oy;
  Bx -= Ox;
  By -= Oy;
  Cx -= Ox;
  Cy -= Oy;

  costemp = 1.0f / sqrtf(1.0f + Ax * Ax / Ay / Ay);
  sintemp = costemp * Ax / Ay;

  ra = Ax * sintemp + Ay * costemp;

  rc = Bx * costemp - By * sintemp;
  rb = Bx * sintemp + By * costemp;
  rd = Cx * costemp - Cy * sintemp;
  re = Cx * sintemp + Cy * costemp;

  if (ra < 0) {
    ra *= -1;
    rb *= -1;
    re *= -1;
  }
}
//对几何信息进行转化
//输入：rAB、rAC：三角形两边长， angle_BAC：AB和AC的夹角(弧度)
//输入：mA、mB、mC：ABC三个的质量
//输出：ra rb rc rd re：位置参数，当刚体三角形质心放置于原点时
// A点放置于(0,ra,0)，B点放置于(rc, rb, 0)，C点放置于(rd, re, 0)
__device__ __host__ void Get_Rabcde_From_SAS(float rAB, float rAC,
                                             float angle_BAC, float mA,
                                             float mB, float mC, float &ra,
                                             float &rb, float &rc, float &rd,
                                             float &re) {
  float mTotal = mA + mB + mC;
  float Ax = 0;
  float Ay = 0;
  float Bx = -rAB;
  float By = 0;

  float costemp = cosf(CONSTANT_Pi - angle_BAC);
  float Cx = rAC * costemp;
  float sintemp = sqrtf(1.0f - costemp * costemp);
  float Cy = rAC * sintemp;

  float Ox = (Bx * mB + Cx * mC) / mTotal;
  float Oy = Cy * mC / mTotal;

  Ax -= Ox;
  Ay -= Oy;
  Bx -= Ox;
  By -= Oy;
  Cx -= Ox;
  Cy -= Oy;

  costemp = 1.0f / sqrtf(1.0f + Ax * Ax / Ay / Ay);
  sintemp = costemp * Ax / Ay;

  ra = Ax * sintemp + Ay * costemp;

  rc = Bx * costemp - By * sintemp;
  rb = Bx * sintemp + By * costemp;
  rd = Cx * costemp - Cy * sintemp;
  re = Cx * sintemp + Cy * costemp;

  if (ra < 0) {
    ra *= -1;
    rb *= -1;
    re *= -1;
  }
}

//核心部分
// 部分参考了Shuichi & Peter: SETTLE: An Analytical Version of the SHAKE and
// RATTLE Algorithm for Rigid Water Models A B C 三个点，O 质心
//输入：rB0 上一步的B原子坐标（A为原点）；rC0 上一步的C原子坐标（A为原点）
// rA1 这一步的A原子坐标（质心为原点） rB1 这一步的B原子坐标（质心为原点）；rC1
// 这一步的C原子坐标（质心为原点） ra rb rc rd
// re：位置参数：当刚体三角形质心放置于原点，A点放置于(0,ra,0)，B点放置于(rc, rb,
// 0)，C点放置于(rd, re, 0)
// mA、mB、mC：ABC三个的质量 dt:步长
// half_exp_gamma_plus_half, exp_gamma: 同simple_constrain
//输出：rA3 这一步限制后的A原子坐标（质心为原点） rB3
//这一步限制后的B原子坐标（质心为 原点） rC3
// 这一步限制后的C原子坐标（质心为原点）vA vB vC 约束后的速度（原位替换） virial
// virial_vector 约束后的维里（原位替换）
__device__ void
SETTLE_DO_TRIANGLE(VECTOR rB0, VECTOR rC0, VECTOR rA1, VECTOR rB1, VECTOR rC1,
                   float ra, float rb, float rc, float rd, float re, float mA,
                   float mB, float mC, float dt, float half_exp_gamma_plus_half,
                   float exp_gamma, VECTOR &rA3, VECTOR &rB3, VECTOR &rC3,
                   VECTOR &vA, VECTOR &vB, VECTOR &vC, VECTOR &virial_vector,
                   int triangle_i) {
  //第0步：构建新坐标系
  // z轴垂直于上一步的BA和BC。 VECTOR ^ VECTOR 是外积
  VECTOR base_vector_z = rB0 ^ rC0;
  // x轴垂直于z轴和这一步的AO
  VECTOR base_vector_x = rA1 ^ base_vector_z;
  // y轴垂直于z轴和x轴
  VECTOR base_vector_y = base_vector_z ^ base_vector_x;
  //归一化
  base_vector_x = rnorm3df(base_vector_x.x, base_vector_x.y, base_vector_x.z) *
                  base_vector_x;
  base_vector_y = rnorm3df(base_vector_y.x, base_vector_y.y, base_vector_y.z) *
                  base_vector_y;
  base_vector_z = rnorm3df(base_vector_z.x, base_vector_z.y, base_vector_z.z) *
                  base_vector_z;

  //第1步：投影至新坐标系
  //     rA0d = {0, 0, 0};
  VECTOR rB0d = {base_vector_x * rB0, base_vector_y * rB0, 0};
  VECTOR rC0d = {base_vector_x * rC0, base_vector_y * rC0, 0};
  VECTOR rA1d = {0, 0, base_vector_z * rA1};
  VECTOR rB1d = {base_vector_x * rB1, base_vector_y * rB1, base_vector_z * rB1};
  VECTOR rC1d = {base_vector_x * rC1, base_vector_y * rC1, base_vector_z * rC1};

  //第2步：绕base_vector_y旋转psi，绕base_vector_x旋转phi得到rX2d
  float sinphi = rA1d.z / ra;
  float cosphi = sqrtf(1.0f - sinphi * sinphi);
  float sinpsi = (rB1d.z - rC1d.z - (rb - re) * sinphi) / ((rd - rc) * cosphi);
  float cospsi = sqrtf(1.0f - sinpsi * sinpsi);

  VECTOR rA2d = {0.0f, ra * cosphi, rA1d.z};
  VECTOR rB2d = {rc * cospsi, rb * cosphi + rc * sinpsi * sinphi, rB1d.z};
  VECTOR rC2d = {rd * cospsi, re * cosphi + rd * sinpsi * sinphi, rC1d.z};

  //第3步：计算辅助变量 alpha、beta、gamma
  float alpha =
      rB2d.x * rB0d.x + rC2d.x * rC0d.x + rB2d.y * rB0d.y + rC2d.y * rC0d.y;
  float beta =
      -rB2d.x * rB0d.y - rC2d.x * rC0d.y + rB2d.y * rB0d.x + rC2d.y * rC0d.x;
  float gamma =
      rB1d.y * rB0d.x - rB1d.x * rB0d.y + rC1d.y * rC0d.x - rC1d.x * rC0d.y;

  //第4步：绕base_vector_z旋转theta
  float temp = alpha * alpha + beta * beta;
  float sintheta = (alpha * gamma - beta * sqrtf(temp - gamma * gamma)) / temp;
  float costheta = sqrt(1.0f - sintheta * sintheta);
  VECTOR rA3d = {-rA2d.y * sintheta, rA2d.y * costheta, rA2d.z};
  VECTOR rB3d = {rB2d.x * costheta - rB2d.y * sintheta,
                 rB2d.x * sintheta + rB2d.y * costheta, rB2d.z};
  VECTOR rC3d = {rC2d.x * costheta - rC2d.y * sintheta,
                 rC2d.x * sintheta + rC2d.y * costheta, rC2d.z};

  //第5步：投影回去
  rA3 = {rA3d.x * base_vector_x.x + rA3d.y * base_vector_y.x +
             rA3d.z * base_vector_z.x,
         rA3d.x * base_vector_x.y + rA3d.y * base_vector_y.y +
             rA3d.z * base_vector_z.y,
         rA3d.x * base_vector_x.z + rA3d.y * base_vector_y.z +
             rA3d.z * base_vector_z.z};

  rB3 = {rB3d.x * base_vector_x.x + rB3d.y * base_vector_y.x +
             rB3d.z * base_vector_z.x,
         rB3d.x * base_vector_x.y + rB3d.y * base_vector_y.y +
             rB3d.z * base_vector_z.y,
         rB3d.x * base_vector_x.z + rB3d.y * base_vector_y.z +
             rB3d.z * base_vector_z.z};

  rC3 = {rC3d.x * base_vector_x.x + rC3d.y * base_vector_y.x +
             rC3d.z * base_vector_z.x,
         rC3d.x * base_vector_x.y + rC3d.y * base_vector_y.y +
             rC3d.z * base_vector_z.y,
         rC3d.x * base_vector_x.z + rC3d.y * base_vector_y.z +
             rC3d.z * base_vector_z.z};

  //第6步：计算约束造成的速度变化和维里变化
  //节约寄存器，把不用的rX1d拿来当delta vX用
  temp = exp_gamma / dt / half_exp_gamma_plus_half;
  rA1d = temp * (rA3 - rA1);
  rB1d = temp * (rB3 - rB1);
  rC1d = temp * (rC3 - rC1);

  vA = vA + rA1d;
  vB = vB + rB1d;
  vC = vC + rC1d;
  //节约寄存器，把不用的rX0d拿来当FX用
  temp = 1.0f / dt / dt / half_exp_gamma_plus_half;
  // rA0d = temp * mA * (rA3 - rA1);
  rB0d = temp * mB * (rB3 - rB1);
  rC0d = temp * mC * (rC3 - rC1);

  virial_vector.x = rB0d.x * rB0.x + rC0d.x * rC0.x;
  virial_vector.y = rB0d.y * rB0.y + rC0d.y * rC0.y;
  virial_vector.z = rB0d.z * rB0.z + rC0d.z * rC0.z;
}

void SETTLE::Initial(CONTROLLER *controller, CONSTRAIN *constrain,
                     float *h_mass, const char *module_name) {
  if (module_name == NULL) {
    strcpy(this->module_name, "settle");
  } else {
    strcpy(this->module_name, module_name);
  }
  if (constrain->constrain_pair_numbers > 0) {
    this->constrain = constrain;
    controller[0].printf("START INITIALIZING SETTLE:\n");
    //遍历搜出constrain里的三角形
    int *linker_numbers = NULL;
    int *linker_atoms = NULL;
    float *link_r = NULL;
    Malloc_Safely((void **)&linker_numbers,
                  sizeof(int) * constrain->atom_numbers);
    Malloc_Safely((void **)&linker_atoms,
                  2 * sizeof(int) * constrain->atom_numbers);
    Malloc_Safely((void **)&link_r,
                  3 * sizeof(float) * constrain->atom_numbers);
    for (int i = 0; i < constrain->atom_numbers; i++) {
      linker_numbers[i] = 0;
    }
    int atom_i, atom_j;
    CONSTRAIN_PAIR pair;
    for (int i = 0; i < constrain->constrain_pair_numbers; i++) {
      pair = constrain->h_constrain_pair[i];
      atom_i = pair.atom_i_serial;
      atom_j = pair.atom_j_serial;

      if (linker_numbers[atom_i] < 2 && linker_numbers[atom_j] < 2) {
        linker_atoms[2 * atom_i + linker_numbers[atom_i]] = atom_j;
        linker_atoms[2 * atom_j + linker_numbers[atom_j]] = atom_i;
        link_r[3 * atom_i + linker_numbers[atom_i]] = pair.constant_r;
        link_r[3 * atom_j + linker_numbers[atom_j]] = pair.constant_r;
        linker_numbers[atom_i]++;
        linker_numbers[atom_j]++;
      } else {
        linker_numbers[atom_i] = 3;
        linker_numbers[atom_j] = 3;
      }
    }

    triangle_numbers = 0;
    pair_numbers = 0;
    for (int i = 0; i < constrain->atom_numbers; i++) {
      if (linker_numbers[i] == 2) {
        atom_i = linker_atoms[2 * i];
        atom_j = linker_atoms[2 * i + 1];
        if (linker_numbers[atom_i] == 2 && linker_numbers[atom_j] == 2 &&
            ((linker_atoms[2 * atom_i] == i &&
              linker_atoms[2 * atom_i + 1] == atom_j) ||
             (linker_atoms[2 * atom_i + 1] == i &&
              linker_atoms[2 * atom_i] == atom_j))) {
          triangle_numbers++;
          linker_numbers[atom_i] = -2;
          linker_numbers[atom_j] = -2;
          if (linker_atoms[2 * atom_i + 1] == atom_j) {
            link_r[3 * i + 2] = link_r[3 * atom_i + 1];
          } else {
            link_r[3 * i + 2] = link_r[3 * atom_i];
          }
        } else {
          linker_numbers[i] = 3;
          linker_numbers[atom_i] = 3;
          linker_numbers[atom_j] = 3;
        }
      } else if (linker_numbers[i] == 1) {
        atom_i = linker_atoms[2 * i];
        if (linker_numbers[atom_i] == 1) {
          pair_numbers++;
          linker_numbers[atom_i] = -1;
        } else {
          linker_numbers[i] = 3;
          linker_numbers[atom_i] = 3;
        }
      }
    }

    controller->printf("    rigid triangle numbers is %d\n", triangle_numbers);
    controller->printf("    rigid pair numbers is %d\n", pair_numbers);
    if (triangle_numbers > 0 || pair_numbers > 0) {
      Malloc_Safely((void **)&h_triangles,
                    sizeof(CONSTRAIN_TRIANGLE) * triangle_numbers);
      Cuda_Malloc_Safely((void **)&d_triangles,
                         sizeof(CONSTRAIN_TRIANGLE) * triangle_numbers);
      Malloc_Safely((void **)&h_pairs, sizeof(CONSTRAIN_PAIR) * pair_numbers);
      Cuda_Malloc_Safely((void **)&d_pairs,
                         sizeof(CONSTRAIN_PAIR) * pair_numbers);

      Cuda_Malloc_Safely((void **)&last_triangle_BA,
                         sizeof(VECTOR) * triangle_numbers);
      Cuda_Malloc_Safely((void **)&last_triangle_CA,
                         sizeof(VECTOR) * triangle_numbers);
      Cuda_Malloc_Safely((void **)&last_pair_AB, sizeof(VECTOR) * pair_numbers);
      Cuda_Malloc_Safely((void **)&virial, sizeof(float));
      Cuda_Malloc_Safely((void **)&virial_vector,
                         sizeof(VECTOR) * (triangle_numbers + pair_numbers));
      int triangle_i = 0;
      int pair_i = 0;
      for (int i = 0; i < constrain->atom_numbers; i++) {
        if (linker_numbers[i] == 2) {
          linker_numbers[i] = -2;
          atom_i = linker_atoms[2 * i];
          atom_j = linker_atoms[2 * i + 1];
          h_triangles[triangle_i].atom_A = i;
          h_triangles[triangle_i].atom_B = atom_i;
          h_triangles[triangle_i].atom_C = atom_j;
          Get_Rabcde_From_SSS(
              link_r[3 * i], link_r[3 * i + 1], link_r[3 * i + 2], h_mass[i],
              h_mass[atom_i], h_mass[atom_j], h_triangles[triangle_i].ra,
              h_triangles[triangle_i].rb, h_triangles[triangle_i].rc,
              h_triangles[triangle_i].rd, h_triangles[triangle_i].re);
          // printf("%d %d %d %f %f %f\n", i, linker_atoms[2 * i],
          // linker_atoms[2 * i + 1], link_r[3 * i], link_r[3 * i + 1], link_r[3
          //* i + 2]);
          triangle_i++;
        }
        if (linker_numbers[i] == 1) {
          atom_j = linker_atoms[2 * i];
          linker_numbers[i] = -1;
          h_pairs[pair_i].atom_i_serial = i;
          h_pairs[pair_i].atom_j_serial = atom_j;
          h_pairs[pair_i].constant_r = link_r[3 * i];
          h_pairs[pair_i].constrain_k = 1.0f / (h_mass[i] + h_mass[atom_j]);
          pair_i++;
        }
      }

      cudaMemcpy(d_triangles, h_triangles,
                 sizeof(CONSTRAIN_TRIANGLE) * triangle_numbers,
                 cudaMemcpyHostToDevice);
      cudaMemcpy(d_pairs, h_pairs, sizeof(CONSTRAIN_PAIR) * pair_numbers,
                 cudaMemcpyHostToDevice);

      //原来的重塑
      int new_constrain_pair_numbers = constrain->constrain_pair_numbers -
                                       3 * triangle_numbers - pair_numbers;
      int new_pair_i = 0;

      CONSTRAIN_PAIR *new_h_constrain_pair = NULL;
      Malloc_Safely((void **)&new_h_constrain_pair,
                    sizeof(CONSTRAIN_PAIR) * new_constrain_pair_numbers);

      for (int i = 0; i < constrain->constrain_pair_numbers; i++) {
        pair = constrain->h_constrain_pair[i];
        atom_i = pair.atom_i_serial;
        if (linker_numbers[atom_i] > 0) {
          new_h_constrain_pair[new_pair_i] = pair;
          new_pair_i++;
        }
      }
      constrain->constrain_pair_numbers = new_constrain_pair_numbers;
      free(constrain->h_constrain_pair);
      cudaFree(constrain->constrain_pair);
      constrain->h_constrain_pair = new_h_constrain_pair;
      Cuda_Malloc_Safely((void **)&constrain->constrain_pair,
                         sizeof(CONSTRAIN_PAIR) * new_constrain_pair_numbers);
      cudaMemcpy(constrain->constrain_pair, constrain->h_constrain_pair,
                 sizeof(CONSTRAIN_PAIR) * new_constrain_pair_numbers,
                 cudaMemcpyHostToDevice);

      controller->printf("    remaining simple constrain pair numbers is %d\n",
                         new_pair_i);
      for (int i = 0; i < constrain->constrain_pair_numbers; i++) {
        pair = constrain->h_constrain_pair[i];
        atom_i = pair.atom_i_serial;
      }
      free(linker_numbers);
      free(linker_atoms);
      free(link_r);
      is_initialized = 1;
      controller[0].printf("END INITIALIZING SETTLE\n\n");
    } else {
      controller[0].printf("SETTLE IS NOT INITIALIZED\n\n");
    }
  } else {
    controller[0].printf("SETTLE IS NOT INITIALIZED\n\n");
  }
}

__global__ void remember_triangle_BA_CA(int triangle_numbers,
                                        CONSTRAIN_TRIANGLE *triangles,
                                        UNSIGNED_INT_VECTOR *uint_crd,
                                        VECTOR scaler, VECTOR *last_triangle_BA,
                                        VECTOR *last_triangle_CA) {
  CONSTRAIN_TRIANGLE triangle;
  for (int triangle_i = blockIdx.x * blockDim.x + threadIdx.x;
       triangle_i < triangle_numbers; triangle_i += blockDim.x * gridDim.x) {
    triangle = triangles[triangle_i];
    last_triangle_BA[triangle_i] = Get_Periodic_Displacement(
        uint_crd[triangle.atom_B], uint_crd[triangle.atom_A], scaler);
    last_triangle_CA[triangle_i] = Get_Periodic_Displacement(
        uint_crd[triangle.atom_C], uint_crd[triangle.atom_A], scaler);
  }
}

__global__ void remember_pair_AB(int pair_numbers, CONSTRAIN_PAIR *pairs,
                                 UNSIGNED_INT_VECTOR *uint_crd, VECTOR scaler,
                                 VECTOR *last_pair_AB) {
  CONSTRAIN_PAIR pair;
  for (int pair_i = blockIdx.x * blockDim.x + threadIdx.x;
       pair_i < pair_numbers; pair_i += blockDim.x * gridDim.x) {
    pair = pairs[pair_i];
    last_pair_AB[pair_i] = Get_Periodic_Displacement(
        uint_crd[pair.atom_j_serial], uint_crd[pair.atom_i_serial], scaler);
  }
}

__global__ void settle_triangle(int triangle_numbers,
                                CONSTRAIN_TRIANGLE *triangles,
                                const float *d_mass, VECTOR *crd,
                                VECTOR box_length, VECTOR *last_triangle_BA,
                                VECTOR *last_triangle_CA, float dt,
                                float exp_gamma, float half_exp_gamma_plus_half,
                                VECTOR *vel, VECTOR *virial_vector) {
  CONSTRAIN_TRIANGLE triangle;
  VECTOR rO;
  VECTOR rA, rB, rC;
  float mA, mB, mC;
  for (int triangle_i = blockIdx.x * blockDim.x + threadIdx.x;
       triangle_i < triangle_numbers; triangle_i += blockDim.x * gridDim.x) {
    triangle = triangles[triangle_i];
    rA = crd[triangle.atom_A];
    rB = Get_Periodic_Displacement(crd[triangle.atom_B], rA, box_length);
    rC = Get_Periodic_Displacement(crd[triangle.atom_C], rA, box_length);
    mA = d_mass[triangle.atom_A];
    mB = d_mass[triangle.atom_B];
    mC = d_mass[triangle.atom_C];

    rO = 1.0f / (mA + mB + mC) * (mB * rB + mC * rC) + rA;
    rA = rA - rO;
    rB = rB + rA;
    rC = rC + rA;

    SETTLE_DO_TRIANGLE(
        last_triangle_BA[triangle_i], last_triangle_CA[triangle_i], rA, rB, rC,
        triangle.ra, triangle.rb, triangle.rc, triangle.rd, triangle.re, mA, mB,
        mC, dt, half_exp_gamma_plus_half, exp_gamma, rA, rB, rC,
        vel[triangle.atom_A], vel[triangle.atom_B], vel[triangle.atom_C],
        virial_vector[triangle_i], triangle_i);

    crd[triangle.atom_A] = rA + rO;
    crd[triangle.atom_B] = rB + rO;
    crd[triangle.atom_C] = rC + rO;
  }
}

__global__ void settle_pair(int pair_numbers, CONSTRAIN_PAIR *pairs,
                            const float *d_mass, VECTOR *crd, VECTOR box_length,
                            VECTOR *last_pair_AB, float dt, float exp_gamma,
                            float half_exp_gamma_plus_half, VECTOR *vel,
                            VECTOR *virial_vector) {
  CONSTRAIN_PAIR pair;
  VECTOR r1, r2, kr2;
  float mA, mB, r0r0, r1r1, r1r2, r2r2, k;
  for (int pair_i = blockIdx.x * blockDim.x + threadIdx.x;
       pair_i < pair_numbers; pair_i += blockDim.x * gridDim.x) {
    pair = pairs[pair_i];

    r1 = Get_Periodic_Displacement(crd[pair.atom_j_serial],
                                   crd[pair.atom_i_serial], box_length);
    r2 = last_pair_AB[pair_i];
    mA = d_mass[pair.atom_i_serial];
    mB = d_mass[pair.atom_j_serial];

    r0r0 = pair.constant_r * pair.constant_r;
    r1r1 = r1 * r1;
    r1r2 = r1 * r2;
    r2r2 = r2 * r2;

    k = (sqrt(r1r2 * r1r2 - r1r1 * r2r2 + r2r2 * r0r0) - r1r2) / r2r2;
    kr2 = k * r2;

    r1 = -mB * pair.constrain_k * kr2;
    kr2 = mA * pair.constrain_k * kr2;

    crd[pair.atom_i_serial] = crd[pair.atom_i_serial] + r1;
    crd[pair.atom_j_serial] = crd[pair.atom_j_serial] + kr2;

    k = exp_gamma / dt / half_exp_gamma_plus_half;
    vel[pair.atom_i_serial] = vel[pair.atom_i_serial] + k * r1;
    vel[pair.atom_j_serial] = vel[pair.atom_j_serial] + k * kr2;

    r1 = k * mB / dt / exp_gamma * kr2;
    virial_vector[pair_i].x = r1.x * r2.x;
    virial_vector[pair_i].y = r1.y * r2.y;
    virial_vector[pair_i].z = r1.z * r2.z;
  }
}

void SETTLE::Remember_Last_Coordinates(UNSIGNED_INT_VECTOR *uint_crd,
                                       VECTOR scaler) {
  if (!is_initialized) {
    return;
  }

  remember_pair_AB<<<64, 1024>>>(pair_numbers, d_pairs, uint_crd, scaler,
                                 last_pair_AB);
  remember_triangle_BA_CA<<<64, 1024>>>(triangle_numbers, d_triangles, uint_crd,
                                        scaler, last_triangle_BA,
                                        last_triangle_CA);
}

__global__ void Sum_Of_Virial_Vector_To_Pressure(int N, VECTOR *virial_vector,
                                                 float *pressure,
                                                 float factor) {
  float virial = 0;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < N;
       i = i + blockDim.x * gridDim.x) {
    virial =
        virial + virial_vector[i].x + virial_vector[i].y + virial_vector[i].z;
  }
  atomicAdd(pressure, virial * factor);
}

void SETTLE::Do_SETTLE(const float *d_mass, VECTOR *crd, VECTOR box_length,
                       VECTOR *vel, int need_pressure, float *d_pressure) {
  if (!is_initialized) {
    return;
  }

  settle_pair<<<64, 64>>>(pair_numbers, d_pairs, d_mass, crd, box_length,
                          last_pair_AB, constrain->dt, constrain->v_factor,
                          constrain->x_factor, vel,
                          virial_vector + triangle_numbers);

  settle_triangle<<<64, 64>>>(triangle_numbers, d_triangles, d_mass, crd,
                              box_length, last_triangle_BA, last_triangle_CA,
                              constrain->dt, constrain->v_factor,
                              constrain->x_factor, vel, virial_vector);

  if (need_pressure) {
    Sum_Of_Virial_Vector_To_Pressure<<<64, 1024>>>(
        triangle_numbers + pair_numbers, virial_vector, d_pressure,
        0.33333f / box_length.x / box_length.y / box_length.z);
  }
}
