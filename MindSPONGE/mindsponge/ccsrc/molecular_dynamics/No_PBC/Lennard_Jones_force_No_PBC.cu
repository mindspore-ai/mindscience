#include "Lennard_Jones_force_No_PBC.cuh"

static __global__ void LJ_Energy_CUDA(const int atom_numbers, const VECTOR *crd, 
    const int *LJ_types, const float *LJ_A, const float *LJ_B, 
    const int *excluded_list_start, const int *excluded_list, const int *excluded_atom_numbers,
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
                float dr_4 = dr_2*dr_2;
                float dr_6 = dr_4*dr_2;
                int type_i = LJ_types[atom_i];
                int type_j = LJ_types[atom_j];
                int type_ij = type_i;
                if (type_i < type_j)
                {
                    type_i = type_j;
                    type_j = type_ij;
                }
                type_ij = ( type_i * (type_i + 1) / 2) + type_j;
                float temp_ene = (0.083333333*LJ_A[type_ij] * dr_6
                    - 0.166666666*LJ_B[type_ij]) * dr_6;
                atomicAdd(&atom_ene[atom_i], temp_ene);
            }
        }
    }
}

static __global__ void LJ_Force_CUDA(const int atom_numbers, const VECTOR *crd,
    const int *LJ_types, const float *LJ_A, const float *LJ_B,
    const int *excluded_list_start, const int *excluded_list, const int *excluded_atom_numbers,
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
                float dr_4 = dr_2*dr_2;
                float dr_6 = dr_4*dr_2;
                float dr_8 = dr_4*dr_4;

                int type_i = LJ_types[atom_i];
                int type_j = LJ_types[atom_j];
                int type_ij = type_i;
                if (type_i < type_j)
                {
                    type_i = type_j;
                    type_j = type_ij;
                }
                type_ij = (type_i * (type_i + 1) / 2) + type_j;
                float frc_abs = (-LJ_A[type_ij] * dr_6
                    + LJ_B[type_ij]) * dr_8;
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

static __global__ void LJ_Force_Energy_CUDA(const int atom_numbers, const VECTOR *crd,
    const int *LJ_types, const float *LJ_A, const float *LJ_B,
    const int *excluded_list_start, const int *excluded_list, const int *excluded_atom_numbers,
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
                float dr_4 = dr_2*dr_2;
                float dr_6 = dr_4*dr_2;
                float dr_8 = dr_4*dr_4;

                int type_i = LJ_types[atom_i];
                int type_j = LJ_types[atom_j];
                int type_ij = type_i;
                if (type_i < type_j)
                {
                    type_i = type_j;
                    type_j = type_ij;
                }
                type_ij = (type_i * (type_i + 1) / 2) + type_j;
                float temp_ene = (0.083333333*LJ_A[type_ij] * dr_6
                    - 0.166666666*LJ_B[type_ij]) * dr_6;
                float frc_abs = (-LJ_A[type_ij] * dr_6
                    + LJ_B[type_ij]) * dr_8;
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

void LENNARD_JONES_NO_PBC_INFORMATION::LJ_Malloc()
{
    Malloc_Safely((void**)&h_LJ_energy_sum, sizeof(float));
    Malloc_Safely((void**)&h_LJ_energy_atom, sizeof(float)*atom_numbers);
    Malloc_Safely((void**)&h_atom_LJ_type, sizeof(int)*atom_numbers);
    Malloc_Safely((void**)&h_LJ_A, sizeof(float)*pair_type_numbers);
    Malloc_Safely((void**)&h_LJ_B, sizeof(float)*pair_type_numbers);

    Cuda_Malloc_Safely((void**)&d_LJ_energy_sum, sizeof(float));
    Cuda_Malloc_Safely((void**)&d_LJ_energy_atom, sizeof(float)*atom_numbers);
    Cuda_Malloc_Safely((void**)&d_atom_LJ_type, sizeof(int)*atom_numbers);
    Cuda_Malloc_Safely((void**)&d_LJ_A, sizeof(float)*pair_type_numbers);
    Cuda_Malloc_Safely((void**)&d_LJ_B, sizeof(float)*pair_type_numbers);
}

void LENNARD_JONES_NO_PBC_INFORMATION::Initial(CONTROLLER *controller, float cutoff, const char *module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "LJ");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    controller[0].printf("START INITIALIZING LENNADR JONES INFORMATION:\n");
    this->cutoff = cutoff;
    if (controller[0].Command_Exist(this->module_name, "in_file"))
    {
        FILE *fp = NULL;
        Open_File_Safely(&fp, controller[0].Command(this->module_name, "in_file"), "r");

        int scanf_ret = fscanf(fp, "%d %d", &atom_numbers, &atom_type_numbers);
        controller[0].printf("    atom_numbers is %d\n", atom_numbers);
        controller[0].printf("    atom_LJ_type_number is %d\n", atom_type_numbers);
        pair_type_numbers = atom_type_numbers * (atom_type_numbers + 1) / 2;
        LJ_Malloc();

        for (int i = 0; i < pair_type_numbers; i++)
        {
            scanf_ret = fscanf(fp, "%f", h_LJ_A + i);
            h_LJ_A[i] *= 12.0f;
        }
        for (int i = 0; i < pair_type_numbers; i++)
        {
            scanf_ret = fscanf(fp, "%f", h_LJ_B + i);
            h_LJ_B[i] *= 6.0f;
        }
        for (int i = 0; i < atom_numbers; i++)
        {
            scanf_ret = fscanf(fp, "%d", h_atom_LJ_type + i);
        }
        fclose(fp);
        Parameter_Host_To_Device();
        is_initialized = 1;
    }
    else if (controller[0].Command_Exist("amber_parm7"))
    {
        Initial_From_AMBER_Parm(controller[0].Command("amber_parm7"), controller[0]);
    }
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller[0].Step_Print_Initial(this->module_name, "%.2f");
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }
    controller[0].printf("END INITIALIZING LENNADR JONES INFORMATION\n\n");
}

void LENNARD_JONES_NO_PBC_INFORMATION::Clear()
{
    if (is_initialized)
    {
        is_initialized = 0;

        free(h_atom_LJ_type);
        cudaFree(d_atom_LJ_type);

        free(h_LJ_A);
        free(h_LJ_B);
        cudaFree(d_LJ_A);
        cudaFree(d_LJ_B);

        free(h_LJ_energy_atom);
        cudaFree(d_LJ_energy_atom);
        cudaFree(d_LJ_energy_sum);

        h_atom_LJ_type = NULL;
        d_atom_LJ_type = NULL;

        h_LJ_A = NULL;
        h_LJ_B = NULL;
        d_LJ_A = NULL;
        d_LJ_B = NULL;

        h_LJ_energy_atom = NULL;
        d_LJ_energy_atom = NULL;
        d_LJ_energy_sum = NULL;
    }
}

void LENNARD_JONES_NO_PBC_INFORMATION::Initial_From_AMBER_Parm(const char *file_name, CONTROLLER controller)
{

    FILE *parm = NULL;
    Open_File_Safely(&parm, file_name, "r");
    controller.printf("    Start reading LJ information from AMBER file:\n");

    while (true)
    {
        char temps[CHAR_LENGTH_MAX];
        char temp_first_str[CHAR_LENGTH_MAX];
        char temp_second_str[CHAR_LENGTH_MAX];
        if (!fgets(temps, CHAR_LENGTH_MAX, parm))
        {
            break;
        }
        if (sscanf(temps, "%s %s", temp_first_str, temp_second_str) != 2)
        {
            continue;
        }
        if (strcmp(temp_first_str, "%FLAG") == 0
            && strcmp(temp_second_str, "POINTERS") == 0)
        {
            char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);

            int scanf_ret = fscanf(parm, "%d\n", &atom_numbers);
            controller.printf("        atom_numbers is %d\n", atom_numbers);
            scanf_ret = fscanf(parm, "%d\n", &atom_type_numbers);
            controller.printf("        atom_LJ_type_number is %d\n", atom_type_numbers);
            pair_type_numbers = atom_type_numbers * (atom_type_numbers + 1) / 2;

            LJ_Malloc();

        }
        if (strcmp(temp_first_str, "%FLAG") == 0
            && strcmp(temp_second_str, "ATOM_TYPE_INDEX") == 0)
        {
            char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            printf("        read atom LJ type index\n");
            int atomljtype;
            for (int i = 0; i<atom_numbers; i = i + 1)
            {
                int scanf_ret = fscanf(parm, "%d\n", &atomljtype);
                h_atom_LJ_type[i] = atomljtype - 1;
            }
        }
        if (strcmp(temp_first_str, "%FLAG") == 0
            && strcmp(temp_second_str, "LENNARD_JONES_ACOEF") == 0)
        {
            char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            printf("        read atom LJ A\n");
            double lin;
            for (int i = 0; i < pair_type_numbers; i = i + 1)
            {
                int scanf_ret = fscanf(parm, "%lf\n", &lin);
                h_LJ_A[i] = (float)12.* lin;
            }
        }
        if (strcmp(temp_first_str, "%FLAG") == 0
            && strcmp(temp_second_str, "LENNARD_JONES_BCOEF") == 0)
        {
            char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            printf("        read atom LJ B\n");
            double lin;
            for (int i = 0; i < pair_type_numbers; i = i + 1)
            {
                int scanf_ret = fscanf(parm, "%lf\n", &lin);
                h_LJ_B[i] = (float)6.* lin;
            }
        }
    }
    controller.printf("    End reading LJ information from AMBER file:\n");
    fclose(parm);
    is_initialized = 1;
    Parameter_Host_To_Device();
}

void LENNARD_JONES_NO_PBC_INFORMATION::Parameter_Host_To_Device()
{
    cudaMemcpy(d_LJ_B, h_LJ_B, sizeof(float)*pair_type_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_LJ_A, h_LJ_A, sizeof(float)*pair_type_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_atom_LJ_type, h_atom_LJ_type, sizeof(int)*atom_numbers, cudaMemcpyHostToDevice);
}


void LENNARD_JONES_NO_PBC_INFORMATION::LJ_Force_With_Atom_Energy(const int atom_numbers, const VECTOR *crd, VECTOR *frc, const int need_atom_energy, float *atom_energy,
    const int *excluded_list_start, const int *excluded_list, const int *excluded_atom_numbers)
{
    if (is_initialized)
    {
        if (!need_atom_energy > 0)
        {
            LJ_Force_CUDA << <{(unsigned int)ceilf((float)atom_numbers / thread_LJ.x), (unsigned int)ceilf((float)atom_numbers / thread_LJ.y)}, { thread_LJ.x, thread_LJ.y } >> >
                (atom_numbers, crd, d_atom_LJ_type, d_LJ_A, d_LJ_B, 
                excluded_list_start, excluded_list, excluded_atom_numbers,
                cutoff * cutoff, frc);
        }
        else
        {
            LJ_Force_Energy_CUDA << <{(unsigned int)ceilf((float)atom_numbers / thread_LJ.x), (unsigned int)ceilf((float)atom_numbers / thread_LJ.y)}, { thread_LJ.x, thread_LJ.y } >> >
                (atom_numbers, crd, d_atom_LJ_type, d_LJ_A, d_LJ_B,
                excluded_list_start, excluded_list, excluded_atom_numbers,
                cutoff * cutoff, atom_energy, frc);
        }
    }
}

float LENNARD_JONES_NO_PBC_INFORMATION::Get_Energy(const VECTOR *crd, const int *excluded_list_start, const int *excluded_list, const int *excluded_atom_numbers, int is_download)
{
    if (is_initialized)
    {
        Reset_List(d_LJ_energy_atom, 0., atom_numbers, 1024);
        LJ_Energy_CUDA << <{(unsigned int)ceilf((float)atom_numbers / thread_LJ.x), (unsigned int)ceilf((float)atom_numbers / thread_LJ.y)}, { thread_LJ.x, thread_LJ.y } >> >
            (atom_numbers, crd, d_atom_LJ_type, d_LJ_A, d_LJ_B,
            excluded_list_start, excluded_list, excluded_atom_numbers,
            cutoff * cutoff, d_LJ_energy_atom);
        Sum_Of_List(d_LJ_energy_atom, d_LJ_energy_sum, atom_numbers);

        if (is_download)
        {
            cudaMemcpy(&h_LJ_energy_sum, this->d_LJ_energy_sum, sizeof(float), cudaMemcpyDeviceToHost);
            return h_LJ_energy_sum;
        }
        else
        {
            return 0;
        }
    }
    return NAN;
}