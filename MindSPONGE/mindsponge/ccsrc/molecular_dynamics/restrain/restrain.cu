#include "restrain.cuh"

//读取rst7
static void Import_Information_From_Rst7(const char *file_name, int *atom_numbers, float *sys_time, VECTOR **crd, VECTOR **vel, VECTOR *box_length, CONTROLLER controller)
{
    FILE *fin = NULL;
    Open_File_Safely(&fin, file_name, "r");
    controller.printf("    Start reading restrain reference coordinate from AMBERFILE\n");
    char lin[CHAR_LENGTH_MAX];
    char *get_ret = fgets(lin, CHAR_LENGTH_MAX, fin);
    get_ret = fgets(lin, CHAR_LENGTH_MAX, fin);
    int has_vel = 0;
    int scanf_ret = sscanf(lin, "%d %f", &atom_numbers[0], &sys_time[0]);
    if (scanf_ret == 2)
    {
        has_vel = 1;
    }
    else
    {
        sys_time[0] = 0.;
    }
    controller.printf("        atom_numbers is %d\n", atom_numbers[0]);
    if (has_vel == 1)
    {
        controller.printf("        system_start_time is %f\n", sys_time[0]);
    }

    VECTOR *h_crd = NULL, *h_vel = NULL;
    Malloc_Safely((void**)&h_crd, sizeof(VECTOR)*atom_numbers[0]);
    Malloc_Safely((void**)&h_vel, sizeof(VECTOR)*atom_numbers[0]);

    Cuda_Malloc_Safely((void**)&crd[0], sizeof(VECTOR)*atom_numbers[0]);
    Cuda_Malloc_Safely((void**)&vel[0], sizeof(VECTOR)*atom_numbers[0]);
    for (int i = 0; i < atom_numbers[0]; i = i + 1)
    {
        scanf_ret = fscanf(fin, "%f %f %f",
            &h_crd[i].x,
            &h_crd[i].y,
            &h_crd[i].z);
    }
    if (has_vel == 1)
    {
        for (int i = 0; i < atom_numbers[0]; i = i + 1)
        {
            scanf_ret = fscanf(fin, "%f %f %f",
                &h_vel[i].x,
                &h_vel[i].y,
                &h_vel[i].z);
        }
    }
    else
    {
        for (int i = 0; i < atom_numbers[0]; i = i + 1)
        {
            h_vel[i].x = 0.0;
            h_vel[i].y = 0.0;
            h_vel[i].z = 0.0;
        }
    }
    scanf_ret = fscanf(fin, "%f %f %f", &box_length[0].x, &box_length[0].y, &box_length[0].z);
    controller.printf("        system size is %f %f %f\n", box_length[0].x, box_length[0].y, box_length[0].z);
    cudaMemcpy(crd[0], h_crd, sizeof(VECTOR)*atom_numbers[0], cudaMemcpyHostToDevice);
    cudaMemcpy(vel[0], h_vel, sizeof(VECTOR)*atom_numbers[0], cudaMemcpyHostToDevice);
    //in some fuck rst7, the coordinates will be extremely bad, so need a full box map
    for (int i = 0; i < 10; i = i + 1)
    {
        Crd_Periodic_Map << <ceilf((float)atom_numbers[0] / 32), 32 >> >
            (atom_numbers[0], crd[0], box_length[0]);
    }
    controller.printf("    End reading restrain reference coordinate from AMBERFILE\n");
    free(h_crd), free(h_vel);
    fclose(fin);
}

static __global__ void restrain_energy(const int restrain_numbers, const int *restrain_list,
    const VECTOR *crd, const VECTOR *crd_ref,
    const float weight, const VECTOR boxlength,
    float *ene)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < restrain_numbers)
    {
        int atom_i = restrain_list[i];
        VECTOR dr = Get_Periodic_Displacement(crd_ref[atom_i], crd[atom_i], boxlength);
        ene[i] = weight * dr * dr;//注意 ene列表只有restrain numbers这么长
    }
}

static __global__ void restrain_force_with_atom_energy_and_virial(const int restrain_numbers, const int *restrain_list, 
    const VECTOR *crd, const VECTOR *crd_ref,
    const float weight, const VECTOR boxlength,
    float *atom_energy, float *atom_virial, VECTOR *frc)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < restrain_numbers)
    {
        int atom_i = restrain_list[i];
        VECTOR dr = Get_Periodic_Displacement(crd_ref[atom_i], crd[atom_i], boxlength);

        VECTOR temp_force = 2 * weight * dr;
        float virial = temp_force * dr;

        atom_energy[atom_i] += 0.5 * virial;
        atom_virial[atom_i] -= virial;
        frc[atom_i] = frc[atom_i] + temp_force;
    }
}

void RESTRAIN_INFORMATION::Initial(CONTROLLER *controller, const int atom_numbers, const VECTOR *crd, const char *module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "restrain");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }

    if (controller[0].Command_Exist(this->module_name, "atom_id"))
    {
        controller[0].printf("START INITIALIZING RESTRAIN:\n");
        this->weight = 20.0f;
        if (controller[0].Command_Exist(this->module_name, "weight"))
            this->weight = atof(controller[0].Command(this->module_name, "weight"));
        controller[0].printf("    %s_weight is %.0f\n", this->module_name, this->weight);

        //用来临时储存最后不需要的内容
        int temp_atom;
        float ref_time;
        int *temp_atom_lists = NULL;
        Malloc_Safely((void**)&temp_atom_lists, sizeof(int)*atom_numbers);
        VECTOR *d_vel = NULL, h_boxlength;
        Cuda_Malloc_Safely((void**)&d_vel, sizeof(VECTOR)*atom_numbers);

        //读参考原子id
        controller[0].printf("    reading %s_atom_id\n", this->module_name);
        this->restrain_numbers = 0;
        FILE *fr = NULL;
        Open_File_Safely(&fr, controller[0].Command(this->module_name, "atom_id"), "r");

        while (fscanf(fr, "%d", &temp_atom) != EOF)
        {
            temp_atom_lists[this->restrain_numbers] = temp_atom;
            this->restrain_numbers++;
        }
        fclose(fr);
        controller[0].printf("        atom_number is %d\n", restrain_numbers);
        Malloc_Safely((void**)&this->h_lists, sizeof(int)*this->restrain_numbers);
        cudaMemcpy(this->h_lists, temp_atom_lists, sizeof(int)*this->restrain_numbers, cudaMemcpyHostToHost);
        Cuda_Malloc_Safely((void**)&this->d_lists, sizeof(int)*this->restrain_numbers);
        cudaMemcpy(this->d_lists, temp_atom_lists, sizeof(int)*this->restrain_numbers, cudaMemcpyHostToDevice);
        Cuda_Malloc_Safely((void**)&this->d_restrain_ene, sizeof(float)*this->restrain_numbers);
        Cuda_Malloc_Safely((void**)&this->d_sum_of_restrain_ene, sizeof(float));

        //读参考原子坐标
        Cuda_Malloc_Safely((void**)&crd_ref, sizeof(VECTOR)*atom_numbers);

        if (controller[0].Command_Exist(this->module_name, "coordinate_in_file"))
        {
            controller[0].printf("    reading restrain reference from %s\n", controller[0].Command(this->module_name, "coordinate_in_file"));
            VECTOR *h_crd = NULL;
            Malloc_Safely((void**)&h_crd, sizeof(VECTOR)*atom_numbers);
            FILE *fp = NULL;
            Open_File_Safely(&fp, controller[0].Command(this->module_name, "coordinate_in_file"), "r");
                        int temp_atom_numbers = 0;
            int scanf_ret = fscanf(fp, "%d", &temp_atom_numbers);
            controller[0].printf("        atom_numbers is %d\n", temp_atom_numbers);
            for (int i = 0; i < atom_numbers; i++)
            {
                scanf_ret = fscanf(fp, "%f %f %f", &h_crd[i].x, &h_crd[i].y, &h_crd[i].z);
            }
            cudaMemcpy(crd_ref, h_crd, sizeof(VECTOR)* atom_numbers, cudaMemcpyHostToDevice);
            free(h_crd);
            fclose(fp);
        }
        else if (controller[0].Command_Exist(this->module_name, "amber_rst7"))
        {
            controller[0].printf("    reading restrain reference from %s\n", controller[0].Command(this->module_name, "amber_rst7"));
            Import_Information_From_Rst7(controller[0].Command(this->module_name, "amber_rst7"),
                &temp_atom, &ref_time, &crd_ref, &d_vel, &h_boxlength, controller[0]);    
        }
        else
        {
            controller[0].printf("    restrain reference coordinate copy from input coordinate\n");
            cudaMemcpy(crd_ref, crd, sizeof(VECTOR)* atom_numbers, cudaMemcpyDeviceToDevice);
        }


        cudaFree(d_vel);
        free(temp_atom_lists);
        is_initialized = 1;

        if (is_initialized && !is_controller_printf_initialized)
        {
            controller[0].Step_Print_Initial(this->module_name, "%.2f");
            is_controller_printf_initialized = 1;
            controller[0].printf("    structure last modify date is %d\n", last_modify_date);
        }
        controller[0].printf("END INITIALIZING RESTRAIN\n\n");
    }
    else
    {
        controller[0].printf("RESTRAIN IS NOT INITIALIZED\n\n");
    }
}

void RESTRAIN_INFORMATION::Restraint(const VECTOR *crd, const VECTOR box_length, float *atom_energy, float *atom_virial, VECTOR *frc)
{
    if (is_initialized)
    {
        restrain_force_with_atom_energy_and_virial << <(unsigned int)ceilf((float)this->restrain_numbers / threads_per_block), threads_per_block >> >
            (this->restrain_numbers, this->d_lists, crd, this->crd_ref,
            this->weight, box_length, atom_energy, atom_virial, frc);
    }
}

float RESTRAIN_INFORMATION::Get_Energy(const VECTOR *crd, const VECTOR box_length, int is_download)
{
    if (is_initialized)
    {
        restrain_energy << <(unsigned int)ceilf((float)this->restrain_numbers / threads_per_block), threads_per_block >> >
            (this->restrain_numbers, this->d_lists, crd, this->crd_ref,
            this->weight, box_length, d_restrain_ene);
        Sum_Of_List(d_restrain_ene, d_sum_of_restrain_ene, restrain_numbers);
        if (is_download)
        {
            cudaMemcpy(&h_sum_of_restrain_ene, d_sum_of_restrain_ene, sizeof(float), cudaMemcpyDeviceToHost);
            return h_sum_of_restrain_ene;
        }
        else
        {
            return 0;
        }
    }
    return NAN;
}



void RESTRAIN_INFORMATION::Clear()
{
    if (is_initialized)
    {
        is_initialized = 0;

        free(h_lists);

        cudaFree(d_lists);
        cudaFree(d_restrain_ene);
        cudaFree(d_sum_of_restrain_ene);
        cudaFree(crd_ref);

        h_lists = NULL;

        d_lists = NULL;
        d_restrain_ene = NULL;
        d_sum_of_restrain_ene = NULL;
        crd_ref = NULL;
    }
}
