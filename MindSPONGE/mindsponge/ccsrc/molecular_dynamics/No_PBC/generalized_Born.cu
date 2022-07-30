#include "generalized_Born.cuh"

static __global__ void Effective_Born_Radii_Factor_CUDA(const int atom_numbers, const VECTOR *crd, const float cutoff_square, const float *self_radius, const float *other_radius, float *effective_radius)
{
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    int atom_j = blockDim.y * blockIdx.y + threadIdx.y;
    if (atom_i < atom_numbers && atom_j < atom_numbers && atom_i != atom_j)
    {
        VECTOR dr = crd[atom_j] - crd[atom_i];
        float dr2 = dr * dr;
        if (dr2 < cutoff_square)
        {
            float R = sqrtf(dr2);
            float self_radii = self_radius[atom_i];
            float other_radii = other_radius[atom_j];
            float inner_distance = R - other_radii;
            float outer_distance = R + other_radii;
            float U = 1;
            float L = 1;
            if (self_radii <= inner_distance)
            {
                L = 1.0 / inner_distance;
                U = 1.0 / outer_distance;
            }
            else if (self_radii < outer_distance)
            {
                L = 1.0 / self_radii;
                U = 1.0 / outer_distance;
            }

            float temp = 0.125 / R * (4 * R * (L - U) + dr2 * (U * U - L * L) + 2 * logf(U / L) + other_radii * other_radii * (L * L - U * U));
            atomicAdd(effective_radius + atom_i, temp);
        }
    }
}

static __global__ void Effective_Born_Radii_CUDA(const int atom_numbers, const float *self_radius, float *effective_radius)
{
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        effective_radius[atom_i] = 1.0 / (1.0 / self_radius[atom_i] - effective_radius[atom_i]);
    }
}

static __global__ void GB_inej_Energy_CUDA(const int atom_numbers, const VECTOR *crd,
    const float *charge, const float *effective_radius, const float epsilon_1_minus_1, const float cutoff_square, float *atom_ene)
{
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    int atom_j = atom_i + 1 + blockDim.y * blockIdx.y + threadIdx.y;
    if (atom_i < atom_numbers && atom_j < atom_numbers)
    {
        VECTOR dr = crd[atom_j] - crd[atom_i];
        float dr2 = dr * dr;
        if (dr2 < cutoff_square)
        {
            float born_radii_ij_square = effective_radius[atom_i] * effective_radius[atom_j];
            float D = expf(-0.25 * dr2 / born_radii_ij_square);
            float f_1 = 1 / sqrtf(dr2 + born_radii_ij_square * D);
            float temp_ene = charge[atom_i] * charge[atom_j] * f_1 * epsilon_1_minus_1;
            atomicAdd(&atom_ene[atom_i], temp_ene);
        }
    }
}

static __global__ void GB_ieqj_Energy_CUDA(const int atom_numbers, const VECTOR *crd,
    const float *charge, const float *effective_radius, const float epsilon_1_minus_1_half, const float cutoff_square, float *atom_ene)
{
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        float f_1 = 1 / effective_radius[atom_i];
        float temp_ene = charge[atom_i];
        temp_ene *= temp_ene;
        temp_ene *= f_1 * epsilon_1_minus_1_half;
        atomicAdd(&atom_ene[atom_i], temp_ene);
    }
}


static __global__ void GB_inej_Force_Energy_CUDA(const int atom_numbers, const VECTOR *crd,
    const float *charge, const float *effective_radius, const float epsilon_1_minus_1, const float cutoff_square,
    VECTOR *frc, float *atom_ene, float *de_da)
{
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    int atom_j = atom_i + 1 + blockDim.y * blockIdx.y + threadIdx.y;
    if (atom_i < atom_numbers && atom_j < atom_numbers)
    {
        VECTOR dr = crd[atom_j] - crd[atom_i];
        float dr2 = dr * dr;
        if (dr2 < cutoff_square)
        {
            SADfloat<3> R(sqrtf(dr2),0);
            SADfloat<3> a1(effective_radius[atom_i], 1);
            SADfloat<3> a2(effective_radius[atom_j], 2);

            SADfloat<3> born_radii_ij_square = a1 * a2;
            SADfloat<3> R2 = R * R;
            SADfloat<3> D = expf(-0.25f * R2 / born_radii_ij_square);
            SADfloat<3> temp_ene = charge[atom_i] * charge[atom_j] * epsilon_1_minus_1 / sqrtf(R2 + born_radii_ij_square * D);
            
            VECTOR temp_frc = -temp_ene.dval[0] / R.val * dr;

            atomicAdd(&frc[atom_j].x, temp_frc.x);
            atomicAdd(&frc[atom_j].y, temp_frc.y);
            atomicAdd(&frc[atom_j].z, temp_frc.z);
            atomicAdd(&frc[atom_i].x, -temp_frc.x);
            atomicAdd(&frc[atom_i].y, -temp_frc.y);
            atomicAdd(&frc[atom_i].z, -temp_frc.z);

            atomicAdd(&de_da[atom_i], temp_ene.dval[1]);
            atomicAdd(&de_da[atom_j], temp_ene.dval[2]);

            atomicAdd(&atom_ene[atom_i], temp_ene.val);
        }
    }
}

static __global__ void GB_ieqj_Force_Energy_CUDA(const int atom_numbers, const VECTOR *crd,
    const float *charge, const float *effective_radius, const float epsilon_1_minus_1_half, const float cutoff_square,
    VECTOR *frc, float *atom_ene, float *de_da)
{
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        float f_1 = 1 / effective_radius[atom_i];
        float temp_ene = charge[atom_i];
        temp_ene *= temp_ene;
        temp_ene *= f_1 * epsilon_1_minus_1_half;

        atomicAdd(&de_da[atom_i], -temp_ene * f_1);
        atomicAdd(&atom_ene[atom_i], temp_ene);
    }
}

static __global__ void GB_accumulate_Force_Energy_CUDA(const int atom_numbers, const VECTOR *crd, const float cutoff_square, const float *self_radius, const float *other_radius,
    const float *effective_raius, const float *de_da, VECTOR *frc)
{
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    int atom_j = blockDim.y * blockIdx.y + threadIdx.y;
    if (atom_i < atom_numbers && atom_j < atom_numbers && atom_i != atom_j)
    {
        VECTOR dr = crd[atom_j] - crd[atom_i];
        float dr2 = dr * dr;
        if (dr2 < cutoff_square)
        {
            SADfloat<1> R(sqrtf(dr2), 0);
            float self_radii = self_radius[atom_i];
            float other_radii = other_radius[atom_j];
            SADfloat<1> inner_distance = R - other_radii;
            SADfloat<1> outer_distance = R + other_radii;
            SADfloat<1> U = 1.0f;
            SADfloat<1> L = 1.0f;
            if (self_radii <= inner_distance.val)
            {
                L = 1.0f / inner_distance;
                U = 1.0f / outer_distance;
            }
            else if (self_radii < outer_distance.val)
            {
                L = 1.0f / self_radii;
                U = 1.0f / outer_distance;
            }

            SADfloat<1> temp = 0.125f / R * (4.0f * R * (L - U) + R * R * (U * U - L * L) + 2.0f * logf(U / L) + other_radii * other_radii * (L * L - U * U));

            float reff = effective_raius[atom_i];
            VECTOR temp_frc = reff * reff * temp.dval[0] * de_da[atom_i] / R.val * dr;
            atomicAdd(&frc[atom_i].x, temp_frc.x);
            atomicAdd(&frc[atom_i].y, temp_frc.y);
            atomicAdd(&frc[atom_i].z, temp_frc.z);
            atomicAdd(&frc[atom_j].x, -temp_frc.x);
            atomicAdd(&frc[atom_j].y, -temp_frc.y);
            atomicAdd(&frc[atom_j].z, -temp_frc.z);
        }
    }
}


void GENERALIZED_BORN_INFORMATION::Malloc()
{
    //Malloc_Safely((void**)&h_Coulomb_energy_sum, sizeof(float));
    Malloc_Safely((void**)&h_GB_energy_atom, sizeof(float)*atom_numbers);
    Malloc_Safely((void**)&h_GB_self_radius, sizeof(float)*atom_numbers);
    Malloc_Safely((void**)&h_GB_other_radius, sizeof(float)*atom_numbers);

    Cuda_Malloc_Safely((void**)&d_GB_energy_sum, sizeof(float));
    Cuda_Malloc_Safely((void**)&d_GB_energy_atom, sizeof(float)*atom_numbers);
    Cuda_Malloc_Safely((void**)&d_GB_self_radius, sizeof(float)*atom_numbers);
    Cuda_Malloc_Safely((void**)&d_GB_other_radius, sizeof(float)*atom_numbers);
    Cuda_Malloc_Safely((void**)&d_GB_effective_radius, sizeof(float)*atom_numbers);
    Cuda_Malloc_Safely((void**)&d_dE_da, sizeof(float)*atom_numbers);
}

void GENERALIZED_BORN_INFORMATION::Initial(CONTROLLER *controller, float cutoff, const char *module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "gb");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    controller[0].printf("START INITIALIZING STANDARD GENERALIZED BORN INFORMATION:\n");
    relative_dielectric_constant = 78.5;
    if (controller->Command_Exist(this->module_name, "epsilon"))
    {
        relative_dielectric_constant = atof(controller->Command(this->module_name, "epsilon"));
    }

    radii_offset = 0.09;
    if (controller->Command_Exist(this->module_name, "radii_offset"))
    {
        radii_offset = atof(controller->Command(this->module_name, "radii_offset"));
    }

    this->cutoff = cutoff;
    radii_cutoff = cutoff;
    if (controller->Command_Exist(this->module_name, "radii_cutoff"))
    {
        radii_cutoff = atof(controller->Command(this->module_name, "radii_cutoff"));
    }

    if (controller->Command_Exist("gb_in_file"))
    {
        FILE *fp;
        Open_File_Safely(&fp, controller->Command("gb_in_file"), "r");
        int scanf_ret = fscanf(fp, "%d", &atom_numbers);
        Malloc();
        for (int i = 0; i < atom_numbers; i++)
        {
            scanf_ret = fscanf(fp, "%f %f", h_GB_self_radius + i, h_GB_other_radius + i);
            h_GB_self_radius[i] -= radii_offset;
            h_GB_other_radius[i] *= h_GB_self_radius[i];
        }
        cudaMemcpy(d_GB_self_radius, h_GB_self_radius, sizeof(float)* atom_numbers, cudaMemcpyHostToDevice);
        cudaMemcpy(d_GB_other_radius, h_GB_other_radius, sizeof(float)* atom_numbers, cudaMemcpyHostToDevice);
    }
    else
    {
        controller->printf("    Error: GB need radii and scaled factor");
        getchar();
        exit(1);
    }

    this->is_initialized = 1;
    
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller[0].Step_Print_Initial(this->module_name, "%.2f");
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }
    controller[0].printf("END INITIALIZING STANDARD GENERALIZED BORN INFORMATION\n\n");
}

void GENERALIZED_BORN_INFORMATION::Clear()
{
    if (is_initialized)
    {
        is_initialized = 0;

        cudaFree(d_GB_energy_sum);

        free(h_GB_self_radius);
        cudaFree(d_GB_self_radius);

        free(h_GB_other_radius);
        cudaFree(d_GB_other_radius);

        free(h_GB_energy_atom);
        cudaFree(d_GB_energy_atom);

        cudaFree(d_GB_effective_radius);
        cudaFree(d_dE_da);

        d_GB_energy_sum = NULL;

        h_GB_energy_atom = NULL;
        d_GB_energy_atom = NULL;

        h_GB_self_radius = NULL;
        d_GB_self_radius = NULL;

        h_GB_other_radius = NULL;
        d_GB_other_radius = NULL;

        d_GB_effective_radius = NULL;
        d_dE_da = NULL;
    }
}

void GENERALIZED_BORN_INFORMATION::Get_Effective_Born_Radius(const VECTOR *crd)
{
    if (is_initialized)
    {
        cudaMemset(d_GB_effective_radius, 0, sizeof(float)*atom_numbers);
        Effective_Born_Radii_Factor_CUDA << <{(unsigned int)ceilf((float)atom_numbers / thread_GB.x), (unsigned int)ceilf((float)atom_numbers / thread_GB.y)}, { thread_GB.x, thread_GB.y } >> >
        (atom_numbers, crd, radii_cutoff * radii_cutoff, d_GB_self_radius, d_GB_other_radius, d_GB_effective_radius);
        Effective_Born_Radii_CUDA << < (unsigned int)ceilf((float)atom_numbers / thread_GB.x / thread_GB.y), thread_GB.x * thread_GB.y >> >
        (atom_numbers, d_GB_self_radius, d_GB_effective_radius);
    }
}

float GENERALIZED_BORN_INFORMATION::Get_Energy(const VECTOR *crd, const float *charge, int is_download)
{
    if (is_initialized)
    {
        cudaMemset(d_GB_energy_atom, 0, sizeof(float)*atom_numbers);
        GB_inej_Energy_CUDA << <{(unsigned int)ceilf((float)atom_numbers / thread_GB.x), (unsigned int)ceilf((float)atom_numbers / thread_GB.y)}, { thread_GB.x, thread_GB.y } >> >(atom_numbers, crd, charge, d_GB_effective_radius, 1.0 / relative_dielectric_constant - 1.0, cutoff * cutoff, d_GB_energy_atom);
        GB_ieqj_Energy_CUDA << < (unsigned int)ceilf((float)atom_numbers / thread_GB.x / thread_GB.y), thread_GB.x * thread_GB.y >> >(atom_numbers, crd, charge, d_GB_effective_radius, 0.5 / relative_dielectric_constant - 0.5, cutoff * cutoff, d_GB_energy_atom);
        Sum_Of_List(d_GB_energy_atom, d_GB_energy_sum, atom_numbers);
        if (is_download)
        {
            cudaMemcpy(&h_GB_energy_sum, this->d_GB_energy_sum, sizeof(float), cudaMemcpyDeviceToHost);
            return h_GB_energy_sum;
        }
        else
        {
            return 0;
        }
    }
    return NAN;
}

void GENERALIZED_BORN_INFORMATION::GB_Force_With_Atom_Energy(const int atom_numbers, const VECTOR *crd, const float *charge, VECTOR *frc, float *atom_energy)
{
    if (is_initialized)
    {
        cudaMemset(d_dE_da, 0, sizeof(float)*atom_numbers);
        GB_inej_Force_Energy_CUDA << <{(unsigned int)ceilf((float)atom_numbers / thread_GB.x), (unsigned int)ceilf((float)atom_numbers / thread_GB.y)}, { thread_GB.x, thread_GB.y } >> >
        (atom_numbers, crd, charge, d_GB_effective_radius, 1.0 / relative_dielectric_constant - 1.0, cutoff * cutoff, frc, atom_energy, d_dE_da);
        GB_ieqj_Force_Energy_CUDA << < (unsigned int)ceilf((float)atom_numbers / thread_GB.x / thread_GB.y), thread_GB.x * thread_GB.y >> >
        (atom_numbers, crd, charge, d_GB_effective_radius, 0.5 / relative_dielectric_constant - 0.5, cutoff * cutoff, frc, atom_energy, d_dE_da);
        GB_accumulate_Force_Energy_CUDA << <{(unsigned int)ceilf((float)atom_numbers / thread_GB.x), (unsigned int)ceilf((float)atom_numbers / thread_GB.y)}, { thread_GB.x, thread_GB.y } >> >
        (atom_numbers, crd, radii_cutoff * radii_cutoff, d_GB_self_radius, d_GB_other_radius, d_GB_effective_radius, d_dE_da, frc);
    }
}
