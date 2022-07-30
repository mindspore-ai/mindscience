#include "Andersen_thermostat.cuh"

static __global__ void MD_Iteration_Leap_Frog_With_Andersen
(const int atom_numbers, const float half_dt, const float dt,
const float *inverse_mass, const float *factor,
VECTOR *vel, VECTOR *crd, VECTOR *frc, VECTOR *acc, VECTOR *random_vel)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < atom_numbers)
    {
        acc[i].x = inverse_mass[i] * frc[i].x;
        acc[i].y = inverse_mass[i] * frc[i].y;
        acc[i].z = inverse_mass[i] * frc[i].z;

        vel[i].x = vel[i].x + dt*acc[i].x;
        vel[i].y = vel[i].y + dt*acc[i].y;
        vel[i].z = vel[i].z + dt*acc[i].z;


        crd[i].x = crd[i].x + half_dt*vel[i].x;
        crd[i].y = crd[i].y + half_dt*vel[i].y;
        crd[i].z = crd[i].z + half_dt*vel[i].z;

        vel[i] = 1 * factor[i] * random_vel[i];

        crd[i].x = crd[i].x + half_dt*vel[i].x;
        crd[i].y = crd[i].y + half_dt*vel[i].y;
        crd[i].z = crd[i].z + half_dt*vel[i].z;

    }
}

static __global__ void MD_Iteration_Leap_Frog_With_Andersen_With_Max_Velocity
(const int atom_numbers, const float half_dt, const float dt, 
const float *inverse_mass, const float *factor,
VECTOR *vel, VECTOR *crd, VECTOR *frc, VECTOR *acc, VECTOR *random_vel, const float max_vel)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    float abs_vel;
    if (i < atom_numbers)
    {
        acc[i].x = inverse_mass[i] * frc[i].x;
        acc[i].y = inverse_mass[i] * frc[i].y;
        acc[i].z = inverse_mass[i] * frc[i].z;

        vel[i].x = vel[i].x + dt*acc[i].x;
        vel[i].y = vel[i].y + dt*acc[i].y;
        vel[i].z = vel[i].z + dt*acc[i].z;


        abs_vel = fminf(1.0, max_vel * rnorm3df(vel[i].x, vel[i].y, vel[i].z));
        vel[i].x = abs_vel* vel[i].x;
        vel[i].y = abs_vel* vel[i].y;
        vel[i].z = abs_vel* vel[i].z;


        crd[i].x = crd[i].x + half_dt*vel[i].x;
        crd[i].y = crd[i].y + half_dt*vel[i].y;
        crd[i].z = crd[i].z + half_dt*vel[i].z;


        vel[i] = 1 * factor[i] * random_vel[i];

        crd[i].x = crd[i].x + half_dt*vel[i].x;
        crd[i].y = crd[i].y + half_dt*vel[i].y;
        crd[i].z = crd[i].z + half_dt*vel[i].z;


    }
}


void ANDERSEN_THERMOSTAT_INFORMATION::Initial(CONTROLLER *controller, float target_temperature, int atom_numbers, float *h_mass, const char *module_name)
{
    controller->printf("START INITIALIZING ANDERSEN THERMOSTAT:\n");
    if (module_name == NULL)
    {
        strcpy(this->module_name, "andersen_thermostat");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }

    controller[0].printf("    target temperature is %.2f K\n", target_temperature);

    int random_seed = time(NULL);
    if (controller[0].Command_Exist(this->module_name, "seed"))
    {
        random_seed = atoi(controller[0].Command(this->module_name, "seed"));
    }
    controller[0].printf("    random seed is %d\n", random_seed);


    float4_numbers = ceil((double)3.* atom_numbers / 4.);
    Cuda_Malloc_Safely((void**)&random_vel, sizeof(float4)* float4_numbers);
    Cuda_Malloc_Safely((void**)&rand_state, sizeof(curandStatePhilox4_32_10_t)* float4_numbers);
    Setup_Rand_Normal_Kernel << <(unsigned int)ceilf((float)float4_numbers / 1024), 1024 >> >
        (float4_numbers, rand_state, random_seed);

    float factor = sqrtf(CONSTANT_kB * target_temperature);
    Cuda_Malloc_Safely((void**)&d_factor, sizeof(float)* atom_numbers);
    Malloc_Safely((void**)&h_factor, sizeof(float)* atom_numbers);


    for (int i = 0; i < atom_numbers; i = i + 1)
    {
        if (h_mass[i] == 0)
            h_factor[i] = 0;
        else
        {
            h_factor[i] = factor*sqrtf(1.0 / h_mass[i]);
        }
    }

    cudaMemcpy(d_factor, h_factor, sizeof(float)* atom_numbers, cudaMemcpyHostToDevice);
    
    update_interval = 500;
    if (controller[0].Command_Exist(this->module_name, "update_interval"))
        update_interval = atoi(controller[0].Command(this->module_name, "update_interval"));
    controller->printf("    The update_interval is %d\n", update_interval);

    //确定是否加上速度上限
    max_velocity = 0;
    if (controller[0].Command_Exist(this->module_name, "velocity_max"))
    {
        sscanf(controller[0].Command(this->module_name, "velocity_max"), "%f", &max_velocity);
        controller[0].printf("    max velocity is %.2f\n", max_velocity);
    }

    is_initialized = 1;
    if (is_initialized && !is_controller_printf_initialized)
    {
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }

    controller->printf("END INITIALIZING ANDERSEN THERMOSTAT\n\n");
}

void ANDERSEN_THERMOSTAT_INFORMATION::MD_Iteration_Leap_Frog(int atom_numbers, VECTOR *vel, VECTOR *crd, VECTOR *frc, VECTOR *acc, float *inverse_mass, float dt)
{
    if (is_initialized)
    {
        Rand_Normal << <(unsigned int)ceilf((float)float4_numbers / 1024), 1024 >> >
            (float4_numbers, rand_state, (float4 *)random_vel);

        if (max_velocity <= 0)
        {    
            MD_Iteration_Leap_Frog_With_Andersen << < (unsigned int)ceilf((float)atom_numbers / 1024), 1024 >> >
                (atom_numbers, dt / 2, dt, inverse_mass, d_factor, vel, crd, frc, acc, random_vel);
        }
        else
        {
            MD_Iteration_Leap_Frog_With_Andersen_With_Max_Velocity << < (unsigned int)ceilf((float)atom_numbers / 1024), 1024 >> >
                (atom_numbers, dt / 2, dt, inverse_mass, d_factor, vel, crd, frc, acc, random_vel, max_velocity);
        }
    }
}
