#include "simple_constrain.cuh"
static __global__ void Constrain_Force_Cycle
(const int constrain_pair_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler,
const CONSTRAIN_PAIR *constrain_pair,const VECTOR *pair_dr,
    VECTOR *test_frc)
{
    int pair_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (pair_i < constrain_pair_numbers)
    {
        CONSTRAIN_PAIR cp = constrain_pair[pair_i];
        float r_1;
        VECTOR dr;
        float frc_abs;
        VECTOR frc_lin;

        dr.x = ((int)(uint_crd[cp.atom_i_serial].uint_x - uint_crd[cp.atom_j_serial].uint_x)) * scaler.x;
        dr.y = ((int)(uint_crd[cp.atom_i_serial].uint_y - uint_crd[cp.atom_j_serial].uint_y)) * scaler.y;
        dr.z = ((int)(uint_crd[cp.atom_i_serial].uint_z - uint_crd[cp.atom_j_serial].uint_z)) * scaler.z;
        r_1=rnorm3df(dr.x, dr.y, dr.z);
        frc_abs = (1. - cp.constant_r*r_1)*cp.constrain_k;


        frc_lin.x = frc_abs*pair_dr[pair_i].x;
        frc_lin.y = frc_abs*pair_dr[pair_i].y;
        frc_lin.z = frc_abs*pair_dr[pair_i].z;


        atomicAdd(&test_frc[cp.atom_j_serial].x, frc_lin.x);
        atomicAdd(&test_frc[cp.atom_j_serial].y, frc_lin.y);
        atomicAdd(&test_frc[cp.atom_j_serial].z, frc_lin.z);

        atomicAdd(&test_frc[cp.atom_i_serial].x, -frc_lin.x);
        atomicAdd(&test_frc[cp.atom_i_serial].y, -frc_lin.y);
        atomicAdd(&test_frc[cp.atom_i_serial].z, -frc_lin.z);
    }
}
static __global__ void Refresh_Uint_Crd(const int atom_numbers, const VECTOR *crd, const VECTOR quarter_crd_to_uint_crd_cof, UNSIGNED_INT_VECTOR *uint_crd, const VECTOR *test_frc,
    const float *mass_inverse,const float half_exp_gamma_plus_half)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        INT_VECTOR tempi;
        VECTOR crd_lin = crd[atom_i];
        VECTOR frc_lin = test_frc[atom_i];
        float mass_lin = mass_inverse[atom_i];

        crd_lin.x = crd_lin.x + half_exp_gamma_plus_half*frc_lin.x*mass_lin;//mass实际为mass的倒数，frc_lin已经乘以dt^2
        crd_lin.y = crd_lin.y + half_exp_gamma_plus_half*frc_lin.y*mass_lin;
        crd_lin.z = crd_lin.z + half_exp_gamma_plus_half*frc_lin.z*mass_lin;

        tempi.int_x = crd_lin.x*quarter_crd_to_uint_crd_cof.x;
        tempi.int_y = crd_lin.y*quarter_crd_to_uint_crd_cof.y;
        tempi.int_z = crd_lin.z*quarter_crd_to_uint_crd_cof.z;

        uint_crd[atom_i].uint_x = tempi.int_x << 2;
        uint_crd[atom_i].uint_y = tempi.int_y << 2;
        uint_crd[atom_i].uint_z = tempi.int_z << 2;
    }
}
static __global__ void Last_Crd_To_dr
(const int constarin_pair_numbers, const VECTOR *atom_crd,
const VECTOR quarter_crd_to_uint_crd_cof, const VECTOR uint_dr_to_dr,
const CONSTRAIN_PAIR *constrain_pair,
VECTOR *pair_dr)
{
    int pair_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (pair_i < constarin_pair_numbers)
    {
        INT_VECTOR tempi;
        INT_VECTOR tempj;
        UNSIGNED_INT_VECTOR uint_crd_i;
        UNSIGNED_INT_VECTOR uint_crd_j;
        CONSTRAIN_PAIR cp = constrain_pair[pair_i];
        VECTOR dr;

        tempi.int_x = atom_crd[cp.atom_i_serial].x*quarter_crd_to_uint_crd_cof.x;
        tempi.int_y = atom_crd[cp.atom_i_serial].y*quarter_crd_to_uint_crd_cof.y;
        tempi.int_z = atom_crd[cp.atom_i_serial].z*quarter_crd_to_uint_crd_cof.z;

        tempj.int_x = atom_crd[cp.atom_j_serial].x*quarter_crd_to_uint_crd_cof.x;
        tempj.int_y = atom_crd[cp.atom_j_serial].y*quarter_crd_to_uint_crd_cof.y;
        tempj.int_z = atom_crd[cp.atom_j_serial].z*quarter_crd_to_uint_crd_cof.z;

        uint_crd_i.uint_x = tempi.int_x << 2;
        uint_crd_i.uint_y = tempi.int_y << 2;
        uint_crd_i.uint_z = tempi.int_z << 2;

        uint_crd_j.uint_x = tempj.int_x << 2;
        uint_crd_j.uint_y = tempj.int_y << 2;
        uint_crd_j.uint_z = tempj.int_z << 2;

        dr.x = ((int)(uint_crd_i.uint_x - uint_crd_j.uint_x)) * uint_dr_to_dr.x;
        dr.y = ((int)(uint_crd_i.uint_y - uint_crd_j.uint_y)) * uint_dr_to_dr.y;
        dr.z = ((int)(uint_crd_i.uint_z - uint_crd_j.uint_z)) * uint_dr_to_dr.z;

        pair_dr[pair_i] = dr;
    }
}
static __global__ void Refresh_Crd_Vel(const int atom_numbers, const float dt_inverse, const float dt, VECTOR *crd, VECTOR *vel, const VECTOR *test_frc,
    const float *mass_inverse, const float exp_gamma, const float half_exp_gamma_plus_half)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        VECTOR crd_lin = crd[atom_i];
        VECTOR frc_lin = test_frc[atom_i];
        VECTOR vel_lin = vel[atom_i];
        float mass_lin = mass_inverse[atom_i];

        frc_lin.x = frc_lin.x*mass_lin;
        frc_lin.y = frc_lin.y*mass_lin;
        frc_lin.z = frc_lin.z*mass_lin;//mass实际为mass的倒数，frc_lin已经乘以dt^2

        crd_lin.x = crd_lin.x + half_exp_gamma_plus_half*frc_lin.x;
        crd_lin.y = crd_lin.y + half_exp_gamma_plus_half*frc_lin.y;
        crd_lin.z = crd_lin.z + half_exp_gamma_plus_half*frc_lin.z;


        vel_lin.x = (vel_lin.x + exp_gamma*frc_lin.x*dt_inverse);
        vel_lin.y = (vel_lin.y + exp_gamma*frc_lin.y*dt_inverse);
        vel_lin.z = (vel_lin.z + exp_gamma*frc_lin.z*dt_inverse);

        crd[atom_i] = crd_lin;
        vel[atom_i] = vel_lin;
    }
}

void SIMPLE_CONSTRAIN::Initial_Simple_Constrain(CONTROLLER *controller, CONSTRAIN *constrain, const char *module_name)
{
    
    //从传入的参数复制基本信息
    this->constrain = constrain;
    if (module_name == NULL)
    {
        strcpy(this->module_name, "simple_constrain");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (constrain->constrain_pair_numbers > 0)
    {
        controller[0].printf("START INITIALIZING SIMPLE CONSTRAIN:\n");
        iteration_numbers = 25;
        if (controller[0].Command_Exist(this->module_name, "iteration_numbers"))
        {
            int scanf_ret = sscanf(controller[0].Command(this->module_name, "iteration_numbers"), "%d", &iteration_numbers);
        }
        controller[0].printf("    constrain iteration step is %d\n", iteration_numbers);

        step_length = 1.0f;
        if (controller[0].Command_Exist(this->module_name, "step_length"))
        {
            int scanf_ret = sscanf(controller[0].Command(this->module_name, "step_length"), "%f", &step_length);
        }
        controller[0].printf("    constrain step length is %.2f\n", step_length);

        Cuda_Malloc_Safely((void**)&constrain_frc, sizeof(VECTOR)*constrain->atom_numbers);
        Cuda_Malloc_Safely((void**)&test_uint_crd, sizeof(UNSIGNED_INT_VECTOR)*constrain->atom_numbers);
        Cuda_Malloc_Safely((void**)&last_pair_dr, sizeof(VECTOR)*constrain->constrain_pair_numbers);
        Cuda_Malloc_Safely((void**)&d_pair_virial, sizeof(float)*constrain->constrain_pair_numbers);
        Cuda_Malloc_Safely((void**)&d_virial, sizeof(float));

        if (is_initialized && !is_controller_printf_initialized)
        {
            is_controller_printf_initialized = 1;
            controller[0].printf("    structure last modify date is %d\n", last_modify_date);
        }
        controller[0].printf("END INITIALIZING SIMPLE CONSTRAIN\n\n");
        is_initialized = 1;
    }
    else
    {
        controller[0].printf("SIMPLE CONSTRAIN IS NOT INITIALIZED\n\n");
    }
    
}

void SIMPLE_CONSTRAIN::Remember_Last_Coordinates(VECTOR *crd, UNSIGNED_INT_VECTOR *uint_crd, VECTOR scaler)
{
    if (is_initialized)
    {
        //获得分子模拟迭代中上一步的距离信息
        Last_Crd_To_dr << <ceilf((float)constrain->constrain_pair_numbers / 128), 128 >> >
            (constrain->constrain_pair_numbers, crd,
            constrain->quarter_crd_to_uint_crd_cof, constrain->uint_dr_to_dr_cof,
            constrain->constrain_pair,
            last_pair_dr);
    }
}

static __global__ void Constrain_Force_Cycle_With_Virial
(const int constrain_pair_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler,
const CONSTRAIN_PAIR *constrain_pair, const VECTOR *pair_dr,
VECTOR *test_frc, float *d_atom_virial)
{
    int pair_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (pair_i < constrain_pair_numbers)
    {
        CONSTRAIN_PAIR cp = constrain_pair[pair_i];
        VECTOR dr0 = pair_dr[pair_i];
        VECTOR dr = Get_Periodic_Displacement(uint_crd[cp.atom_i_serial], uint_crd[cp.atom_j_serial], scaler);
        float r_1 = rnorm3df(dr.x, dr.y, dr.z);
        float frc_abs = (1. - cp.constant_r*r_1)*cp.constrain_k;
        VECTOR frc_lin = frc_abs * dr0;
        d_atom_virial[pair_i] -= frc_lin * dr0;
        //atomicAdd(d_atom_virial + cp.atom_j_serial, -frc_lin * dr0);

        atomicAdd(&test_frc[cp.atom_j_serial].x, frc_lin.x);
        atomicAdd(&test_frc[cp.atom_j_serial].y, frc_lin.y);
        atomicAdd(&test_frc[cp.atom_j_serial].z, frc_lin.z);

        atomicAdd(&test_frc[cp.atom_i_serial].x, -frc_lin.x);
        atomicAdd(&test_frc[cp.atom_i_serial].y, -frc_lin.y);
        atomicAdd(&test_frc[cp.atom_i_serial].z, -frc_lin.z);
    }
}

static __global__ void pressure_fix(float *pressure, float *virial, float factor)
{
    pressure[0] += factor * virial[0];
}

void SIMPLE_CONSTRAIN::Constrain
(VECTOR *crd, VECTOR *vel, const float *mass_inverse, const float *d_mass, VECTOR box_length, int need_pressure, float *d_pressure)
{
    if (is_initialized)
    {
        //清空约束力和维里
        Reset_List << <ceilf((float)3.*constrain->atom_numbers / 128), 128 >> >
            (3 * constrain->atom_numbers, (float*)constrain_frc, 0.);
        if (need_pressure > 0)
        {
            Reset_List << <ceilf((float)constrain->constrain_pair_numbers / 1024.0f), 1024 >> >(constrain->constrain_pair_numbers, d_pair_virial, 0.0f);
            Reset_List << <1, 1 >> >(1, d_virial, 0.0f);
        }
        for (int i = 0; i < iteration_numbers; i = i + 1)
        {
        
            Refresh_Uint_Crd << <ceilf((float)constrain->atom_numbers / 128), 128 >> >
                (constrain->atom_numbers, crd, constrain->quarter_crd_to_uint_crd_cof, test_uint_crd, constrain_frc,
                mass_inverse, constrain->x_factor);

            if (need_pressure > 0)
            {
                Constrain_Force_Cycle_With_Virial << <ceilf((float)constrain->constrain_pair_numbers / 128), 128 >> >
                    (constrain->constrain_pair_numbers, test_uint_crd, constrain->uint_dr_to_dr_cof,
                    constrain->constrain_pair, last_pair_dr,
                    constrain_frc, d_pair_virial);
            }
            else
            {
                Constrain_Force_Cycle << <ceilf((float)constrain->constrain_pair_numbers / 128), 128 >> >
                    (constrain->constrain_pair_numbers, test_uint_crd, constrain->uint_dr_to_dr_cof,
                    constrain->constrain_pair, last_pair_dr,
                    constrain_frc);
            }
        }
        if (need_pressure > 0)
        {
            Sum_Of_List << <1, 1024 >> >(constrain->constrain_pair_numbers, d_pair_virial, d_virial);
            pressure_fix << <1, 1 >> >(d_pressure, d_virial, 1 / constrain->dt / constrain->dt / 3.0 / constrain->volume);
        }
    
        Refresh_Crd_Vel << <ceilf((float)constrain->atom_numbers / 128), 128 >> >
            (constrain->atom_numbers, constrain->dt_inverse, constrain->dt, crd, vel, constrain_frc,
            mass_inverse, constrain->v_factor, constrain->x_factor);
    }
}


void SIMPLE_CONSTRAIN::Clear()
{
    if (is_initialized)
    {
        is_initialized = 0;

        cudaFree(last_pair_dr);
        last_pair_dr = NULL;

        cudaFree(constrain_frc);
        constrain_frc = NULL;

        cudaFree(test_uint_crd);
        test_uint_crd = NULL;

        cudaFree(d_pair_virial);
        d_pair_virial = NULL;

        cudaFree(d_virial);
        d_virial = NULL;
    }
}
