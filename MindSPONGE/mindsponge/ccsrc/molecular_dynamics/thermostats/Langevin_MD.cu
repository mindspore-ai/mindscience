#include "Langevin_MD.cuh"

static __global__ void MD_Iteration_Leap_Frog_With_Langevin(const int atom_numbers, const float dt, const float *inverse_mass, 
    const float gamma_ln, const float *sigma_mass,
    VECTOR *vel, VECTOR *crd, VECTOR *frc, VECTOR *acc, VECTOR *random_frc)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < atom_numbers)
    {
        
        acc[i].x = inverse_mass[i] * frc[i].x;
        acc[i].y = inverse_mass[i] * frc[i].y;
        acc[i].z = inverse_mass[i] * frc[i].z;

        random_frc[i].x = sigma_mass[i] * random_frc[i].x;
        random_frc[i].y = sigma_mass[i] * random_frc[i].y;
        random_frc[i].z = sigma_mass[i] * random_frc[i].z;
        acc[i].x = acc[i].x - gamma_ln*vel[i].x + random_frc[i].x;
        acc[i].y = acc[i].y - gamma_ln*vel[i].y + random_frc[i].y;
        acc[i].z = acc[i].z - gamma_ln*vel[i].z + random_frc[i].z;

        

        vel[i].x = vel[i].x + dt*acc[i].x;
        vel[i].y = vel[i].y + dt*acc[i].y;
        vel[i].z = vel[i].z + dt*acc[i].z;

        
        
        crd[i].x = crd[i].x + dt*vel[i].x;
        crd[i].y = crd[i].y + dt*vel[i].y;
        crd[i].z = crd[i].z + dt*vel[i].z;

    }
}

static __global__ void MD_Iteration_Leap_Frog_With_Langevin_With_Max_Velocity(const int atom_numbers, const float dt, const float *inverse_mass,
    const float gamma_ln, const float *sigma_mass,
    VECTOR *vel, VECTOR *crd, VECTOR *frc, VECTOR *acc, VECTOR *random_frc, const float max_velocity)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < atom_numbers)
    {
        VECTOR acc_i = inverse_mass[i] * frc[i] - gamma_ln * vel[i] + sigma_mass[i] * random_frc[i];
        VECTOR vel_i = vel[i] + dt * acc_i;
        vel[i] = Make_Vector_Not_Exceed_Value(vel_i, max_velocity);
        crd[i] = crd[i] + dt * vel[i];

    }
}

void Langevin_MD_INFORMATION::Initial(CONTROLLER *controller,const int atom_numbers, const float target_temperature,const float *h_mass, const char *module_name)
{
    controller[0].printf("START INITIALIZING LANGEVIN DYNAMICS:\n");
    if (module_name == NULL)
    {
        strcpy(this->module_name, "langevin");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    
    controller[0].printf("    atom_numbers is %d\n", atom_numbers);
    this->atom_numbers = atom_numbers;
    this->target_temperature = target_temperature;
    float *h_mass_temp = NULL;
    Malloc_Safely((void**)&h_mass_temp, sizeof(float)*atom_numbers);
    cudaMemcpy(h_mass_temp, h_mass, sizeof(float)*atom_numbers,cudaMemcpyHostToHost);

    gamma_ln = 1.0f;
    if (controller[0].Command_Exist(this->module_name, "gamma"))
    {
        gamma_ln = atof(controller[0].Command(this->module_name, "gamma"));
    }

    int random_seed = rand();
    if (controller[0].Command_Exist(this->module_name, "seed"))
    {
        random_seed = atoi(controller[0].Command(this->module_name, "seed"));
    }

    controller[0].printf("    target temperature is %.2f K\n", target_temperature);
    controller[0].printf("    friction coefficient is %.2f ps^-1\n", gamma_ln);
    controller[0].printf("    random seed is %d\n", random_seed);
        
    dt = 0.001;
    if (controller[0].Command_Exist("dt"))
        dt = atof(controller[0].Command("dt"));
    dt *= CONSTANT_TIME_CONVERTION;
    half_dt = 0.5 * dt;




    float4_numbers = ceil((double)3.* atom_numbers / 4.);
    Cuda_Malloc_Safely((void**)&random_force, sizeof(float4)* float4_numbers);
    Cuda_Malloc_Safely((void**)&rand_state, sizeof(curandStatePhilox4_32_10_t)* float4_numbers);

    Setup_Rand_Normal_Kernel << <(unsigned int)ceilf((float)float4_numbers / threads_per_block), threads_per_block >> >
        (float4_numbers, rand_state, random_seed);

    gamma_ln = gamma_ln / CONSTANT_TIME_CONVERTION;//单位换算
    sigma_ln = sqrtf(2.* gamma_ln * CONSTANT_kB * target_temperature / dt);
    Cuda_Malloc_Safely((void**)&d_sigma_mass, sizeof(float)* atom_numbers);
    Malloc_Safely((void**)&h_sigma_mass, sizeof(float)* atom_numbers);
    for (int i = 0; i < atom_numbers; i = i + 1)
    {
        if (h_mass_temp[i] == 0)
            h_sigma_mass[i] = 0;
        else
        {
            h_sigma_mass[i] = sigma_ln*sqrtf(1.0 / h_mass_temp[i]);
        }
            
    }
    cudaMemcpy(d_sigma_mass, h_sigma_mass, sizeof(float)* atom_numbers, cudaMemcpyHostToDevice);
    Cuda_Malloc_Safely((void**)&d_mass_inverse, sizeof(float)*atom_numbers);
    for (int i = 0; i < atom_numbers; i = i + 1)
    {
        if (h_mass_temp[i] != 0)
        {
            h_mass_temp[i] = 1.0 / h_mass_temp[i];
        }
    }
    cudaMemcpy(d_mass_inverse, h_mass_temp, sizeof(float)* atom_numbers, cudaMemcpyHostToDevice);
    //确定是否加上速度上限
    max_velocity = 0;
    if (controller[0].Command_Exist(this->module_name, "velocity_max"))
    {
        sscanf(controller[0].Command(this->module_name, "velocity_max"), "%f", &max_velocity);
        controller[0].printf("    max velocity is %.2f\n", max_velocity);
    }

    free(h_mass_temp);
    
    is_initialized = 1;
    if (is_initialized && !is_controller_printf_initialized)
    {
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }
    controller[0].printf("END INITIALIZING LANGEVIN DYNAMICS\n\n");
}

void Langevin_MD_INFORMATION::MD_Iteration_Leap_Frog(VECTOR *frc, VECTOR *crd, VECTOR *vel, VECTOR *acc)
{
    if (is_initialized)
    {
        Rand_Normal << <(unsigned int)ceilf((float)float4_numbers / threads_per_block), threads_per_block >> >
            (float4_numbers, rand_state, (float4 *)random_force);

        if (max_velocity <= 0)
        {
            MD_Iteration_Leap_Frog_With_Langevin << <(unsigned int)ceilf((float)atom_numbers / threads_per_block), threads_per_block >> >
                (atom_numbers, dt, d_mass_inverse, gamma_ln, d_sigma_mass, vel, crd, frc, acc, random_force);
        }
        else
        {
            MD_Iteration_Leap_Frog_With_Langevin_With_Max_Velocity << <(unsigned int)ceilf((float)atom_numbers / threads_per_block), threads_per_block >> >
                (atom_numbers, dt, d_mass_inverse, gamma_ln, d_sigma_mass, vel, crd, frc, acc, random_force, max_velocity);
        }
    }
}

void Langevin_MD_INFORMATION::Clear()
{
    if (is_initialized)
    {
        is_initialized = 0;
        cudaFree(rand_state);
        cudaFree(random_force);
        free(h_sigma_mass);
        cudaFree(d_sigma_mass);
        cudaFree(d_mass_inverse);
        rand_state = NULL;
        random_force = NULL;
        h_sigma_mass = NULL;
        d_sigma_mass = NULL;
        d_mass_inverse = NULL;
    }
}