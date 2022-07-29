#include "Coulomb_Force_No_PBC.cuh"

static __global__ void Coulomb_Energy_CUDA(const int atom_numbers, const VECTOR *crd,
    const float *charge, const int *excluded_list_start, const int *excluded_list, const int *excluded_atom_numbers,
    const float cutoff_square, float *atom_ene)
{
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    int atom_j = atom_i + 1 + blockDim.y * blockIdx.y + threadIdx.y;
    if (atom_i < atom_numbers && atom_j < atom_numbers)
    {
        int tocal = 1;
        const int *start = excluded_list + excluded_list_start[atom_i];
        for (int k = 0; tocal == 1 && k < excluded_atom_numbers[atom_i]; k += 1)
        {
            if (start[k] == atom_j)
                tocal = 0;
        }
        if (tocal == 1)
        {
            VECTOR dr = crd[atom_j] - crd[atom_i];
            float dr2 = dr * dr;
            if (dr2 < cutoff_square)
            {
                float dr_2 = 1. / dr2;
                float dr_1 = sqrtf(dr_2);
                float temp_ene = charge[atom_i] * charge[atom_j] * dr_1;
                atomicAdd(&atom_ene[atom_i], temp_ene);
            }
        }
    }
}

static __global__ void Coulomb_Force_CUDA(const int atom_numbers, const VECTOR *crd,
    const float *charge, const int *excluded_list_start, const int *excluded_list, const int *excluded_atom_numbers,
    const float cutoff_square, VECTOR *frc)
{
    int atom_i = blockDim.y * blockIdx.y + threadIdx.y;
    int atom_j = atom_i + 1 + blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers && atom_j < atom_numbers)
    {
        int tocal = 1;
        const int *start = excluded_list + excluded_list_start[atom_i];
        for (int k = 0; tocal == 1 && k < excluded_atom_numbers[atom_i]; k += 1)
        {
            if (start[k] == atom_j)
                tocal = 0;
        }
        if (tocal == 1)
        {
            VECTOR dr = crd[atom_j] - crd[atom_i];
            float dr2 = dr * dr;
            if (dr2 < cutoff_square)
            {
                float dr_2 = 1. / dr2;
                float dr_1 = sqrtf(dr_2);
                float dr_3 = dr_1 * dr_2;
                float chargeij = charge[atom_i] * charge[atom_j];
                float frc_abs = -chargeij * dr_3;
                VECTOR temp_frc = frc_abs * dr;

                atomicAdd(&frc[atom_j].x, -temp_frc.x);
                atomicAdd(&frc[atom_j].y, -temp_frc.y);
                atomicAdd(&frc[atom_j].z, -temp_frc.z);
                atomicAdd(&frc[atom_i].x, temp_frc.x);
                atomicAdd(&frc[atom_i].y, temp_frc.y);
                atomicAdd(&frc[atom_i].z, temp_frc.z);
            }
        }
    }
}

static __global__ void Coulomb_Force_Energy_CUDA(const int atom_numbers, const VECTOR *crd,
    const float *charge, const int *excluded_list_start, const int *excluded_list, const int *excluded_atom_numbers,
    const float cutoff_square, float *atom_ene, VECTOR *frc)
{
    int atom_i = blockDim.y * blockIdx.y + threadIdx.y;
    int atom_j = atom_i + 1 + blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers && atom_j < atom_numbers)
    {
        int tocal = 1;
        const int *start = excluded_list + excluded_list_start[atom_i];
        for (int k = 0; tocal == 1 && k < excluded_atom_numbers[atom_i]; k += 1)
        {
            if (start[k] == atom_j)
                tocal = 0;
        }
        if (tocal == 1)
        {
            VECTOR dr = crd[atom_j] - crd[atom_i];
            float dr2 = dr * dr;
            if (dr2 < cutoff_square)
            {
                float dr_2 = 1. / dr2;
                float dr_1 = sqrtf(dr_2);
                float dr_3 = dr_1 * dr_2;
                float chargeij = charge[atom_i] * charge[atom_j];
                float temp_ene = chargeij * dr_1;
                float frc_abs = -chargeij * dr_3;
                VECTOR temp_frc = frc_abs * dr;

                atomicAdd(&frc[atom_j].x, -temp_frc.x);
                atomicAdd(&frc[atom_j].y, -temp_frc.y);
                atomicAdd(&frc[atom_j].z, -temp_frc.z);
                atomicAdd(&frc[atom_i].x, temp_frc.x);
                atomicAdd(&frc[atom_i].y, temp_frc.y);
                atomicAdd(&frc[atom_i].z, temp_frc.z);

                atomicAdd(&atom_ene[atom_i], temp_ene);
            }
        }
    }
}

void COULOMB_FORCE_NO_PBC_INFORMATION::Malloc()
{
    //Malloc_Safely((void**)&h_Coulomb_energy_sum, sizeof(float));
    Malloc_Safely((void**)&h_Coulomb_energy_atom, sizeof(float)*atom_numbers);

    Cuda_Malloc_Safely((void**)&d_Coulomb_energy_sum, sizeof(float));
    Cuda_Malloc_Safely((void**)&d_Coulomb_energy_atom, sizeof(float)*atom_numbers);
}

void COULOMB_FORCE_NO_PBC_INFORMATION::Initial(CONTROLLER *controller, int atom_numbers, float cutoff, const char *module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "Coulomb");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    controller[0].printf("START INITIALIZING COULOMB INFORMATION:\n");
    this->cutoff = cutoff;
    this->atom_numbers = atom_numbers;
    this->is_initialized = 1;
    Malloc();
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller[0].Step_Print_Initial(this->module_name, "%.2f");
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }
    controller[0].printf("END INITIALIZING COULOMB INFORMATION\n\n");
}

void COULOMB_FORCE_NO_PBC_INFORMATION::Clear()
{
    if (is_initialized)
    {
        is_initialized = 0;

        cudaFree(d_Coulomb_energy_sum);

        free(h_Coulomb_energy_atom);
        cudaFree(d_Coulomb_energy_atom);

        d_Coulomb_energy_sum = NULL;

        h_Coulomb_energy_atom = NULL;
        d_Coulomb_energy_atom = NULL;
    }
}


void COULOMB_FORCE_NO_PBC_INFORMATION::Coulomb_Force_With_Atom_Energy(const int atom_numbers, const VECTOR *crd, const float *charge, VECTOR *frc, const int need_atom_energy, float *atom_energy,
    const int *excluded_list_start, const int *excluded_list, const int *excluded_atom_numbers)
{
    if (is_initialized)
    {
        if (!need_atom_energy > 0)
        {
            Coulomb_Force_CUDA << <{(unsigned int)ceilf((float)atom_numbers / thread_Coulomb.x), (unsigned int)ceilf((float)atom_numbers / thread_Coulomb.y)}, { thread_Coulomb.x, thread_Coulomb.y } >> >
                (atom_numbers, crd, charge,    excluded_list_start, excluded_list, excluded_atom_numbers,
                cutoff * cutoff, frc);
        }
        else
        {
            Coulomb_Force_Energy_CUDA << <{(unsigned int)ceilf((float)atom_numbers / thread_Coulomb.x), (unsigned int)ceilf((float)atom_numbers / thread_Coulomb.y)}, { thread_Coulomb.x, thread_Coulomb.y } >> >
                (atom_numbers, crd, charge, excluded_list_start, excluded_list, excluded_atom_numbers,
                cutoff * cutoff, atom_energy, frc);
        }
    }
}

float COULOMB_FORCE_NO_PBC_INFORMATION::Get_Energy(const VECTOR *crd, const float *charge, const int *excluded_list_start, const int *excluded_list, const int *excluded_atom_numbers, int is_download)
{
    if (is_initialized)
    {
        Reset_List(d_Coulomb_energy_atom, 0., atom_numbers, 1024);
        Coulomb_Energy_CUDA << <{(unsigned int)ceilf((float)atom_numbers / thread_Coulomb.x), (unsigned int)ceilf((float)atom_numbers / thread_Coulomb.y)}, { thread_Coulomb.x, thread_Coulomb.y } >> >
            (atom_numbers, crd, charge, excluded_list_start, excluded_list, excluded_atom_numbers,
            cutoff * cutoff, d_Coulomb_energy_atom);
        Sum_Of_List(d_Coulomb_energy_atom, d_Coulomb_energy_sum, atom_numbers);

        if (is_download)
        {
            cudaMemcpy(&h_Coulomb_energy_sum, this->d_Coulomb_energy_sum, sizeof(float), cudaMemcpyDeviceToHost);
            return h_Coulomb_energy_sum;
        }
        else
        {
            return 0;
        }
    }
    return NAN;
}
