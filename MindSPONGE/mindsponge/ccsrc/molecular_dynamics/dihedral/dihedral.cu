#include "dihedral.cuh"

static __global__ void Dihedral_Energy_CUDA(const int dihedral_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler,
    const int *atom_a, const int *atom_b, const int *atom_c, const int *atom_d, const int *ipn, const float *pk, const float *gamc, const float *gams, const float *pn, float *ene)
{
    int dihedral_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (dihedral_i < dihedral_numbers)
    {
        int atom_i = atom_a[dihedral_i];
        int atom_j = atom_b[dihedral_i];
        int atom_k = atom_c[dihedral_i];
        int atom_l = atom_d[dihedral_i];

//        int temp_ipn = ipn[dihedral_i];

        float temp_pk = pk[dihedral_i];
        float temp_pn = pn[dihedral_i];
        float temp_gamc = gamc[dihedral_i];
        float temp_gams = gams[dihedral_i];

        VECTOR drij = Get_Periodic_Displacement(uint_crd[atom_i], uint_crd[atom_j], scaler);
        VECTOR drkj = Get_Periodic_Displacement(uint_crd[atom_k], uint_crd[atom_j], scaler);
        VECTOR drkl = Get_Periodic_Displacement(uint_crd[atom_k], uint_crd[atom_l], scaler);

        VECTOR r1 = drij ^ drkj;
        VECTOR r2 = drkl ^ drkj;

        float r1_1 = rnorm3df(r1.x, r1.y, r1.z);
        float r2_1 = rnorm3df(r2.x, r2.y, r2.z);
        float r1_1_r2_1 = r1_1 * r2_1;

        float phi = r1 * r2 * r1_1_r2_1;
        phi = fmaxf(-0.999999, fminf(phi, 0.999999));
        phi = acosf(phi);

        float sign = (r2 ^ r1) * drkj;
        phi=copysignf(phi, sign);

        phi = CONSTANT_Pi - phi;

        float nphi = temp_pn * phi;

        float cos_nphi = cosf(nphi);
        float sin_nphi = sinf(nphi);

        ene[dihedral_i] = (temp_pk + cos_nphi * temp_gamc + sin_nphi * temp_gams);
    }
}

static __global__ void Dihedral_Force_With_Atom_Energy_CUDA(const int dihedral_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler,
    const int *atom_a, const int *atom_b, const int *atom_c, const int *atom_d, const int *ipn, const float *pk, const float *gamc, const float *gams, const float *pn, VECTOR *frc,
    float *ene)
{
    int dihedral_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (dihedral_i < dihedral_numbers)
    {
        int atom_i = atom_a[dihedral_i];
        int atom_j = atom_b[dihedral_i];
        int atom_k = atom_c[dihedral_i];
        int atom_l = atom_d[dihedral_i];

        int temp_ipn = ipn[dihedral_i];

        float temp_pk = pk[dihedral_i];
        float temp_pn = pn[dihedral_i];
        float temp_gamc = gamc[dihedral_i];
        float temp_gams = gams[dihedral_i];

        VECTOR drij = Get_Periodic_Displacement(uint_crd[atom_i], uint_crd[atom_j], scaler);
        VECTOR drkj = Get_Periodic_Displacement(uint_crd[atom_k], uint_crd[atom_j], scaler);
        VECTOR drkl = Get_Periodic_Displacement(uint_crd[atom_k], uint_crd[atom_l], scaler);

        VECTOR r1 = drij ^ drkj;
        VECTOR r2 = drkl ^ drkj;

        float r1_1 = rnorm3df(r1.x, r1.y, r1.z);
        float r2_1 = rnorm3df(r2.x, r2.y, r2.z);
        float r1_2 = r1_1 * r1_1;
        float r2_2 = r2_1 * r2_1;
        float r1_1_r2_1 = r1_1 * r2_1;
                //PHI, pay attention to the var NAME
        float phi = r1 * r2 * r1_1_r2_1;
        phi = fmaxf(-0.999999, fminf(phi, 0.999999));
        phi = acosf(phi);

        float sign = (r2 ^ r1) * drkj;
        phi=copysignf(phi, sign);

        phi = CONSTANT_Pi - phi;

        float nphi = temp_pn * phi;

        float cos_phi = cosf(phi);
        float sin_phi = sinf(phi);
        float cos_nphi = cosf(nphi);
        float sin_nphi = sinf(nphi);

                //Here and following var name "phi" corespongding to the declaration of phi
                //aka, the var with the comment line "PHI, pay attention to the var NAME" 
                //The real dihedral = Pi - ArcCos(so-called "phi")
                //d(real dihedral) = 1/sin(real dihedral) * d(so-called  "phi")
        float dE_dphi;
        if (fabsf(sin_phi) < 1e-6)
        {
            temp_ipn *= (((temp_ipn - 1) & 1) ^ 1);
            dE_dphi = temp_gamc * (temp_pn - temp_ipn + temp_ipn * cos_phi);
        }
        else
            dE_dphi = temp_pn * (temp_gamc * sin_nphi - temp_gams * cos_nphi) / sin_phi;

        VECTOR dphi_dr1 = r1_1_r2_1 * r2 + cos_phi * r1_2 * r1;
        VECTOR dphi_dr2 = r1_1_r2_1 * r1 + cos_phi * r2_2 * r2;

        VECTOR dE_dri = dE_dphi * drkj ^ dphi_dr1;
        VECTOR dE_drl = dE_dphi * dphi_dr2 ^ drkj;
        VECTOR dE_drj_part = dE_dphi * ((drij ^ dphi_dr1) + (drkl ^ dphi_dr2));

        VECTOR fi = dE_dri;
        VECTOR fj = dE_drj_part - dE_dri;
        VECTOR fk = -dE_drl - dE_drj_part;
        VECTOR fl = dE_drl;

        atomicAdd(&frc[atom_i].x, fi.x);
        atomicAdd(&frc[atom_i].y, fi.y);
        atomicAdd(&frc[atom_i].z, fi.z);
        atomicAdd(&frc[atom_j].x, fj.x);
        atomicAdd(&frc[atom_j].y, fj.y);
        atomicAdd(&frc[atom_j].z, fj.z);
        atomicAdd(&frc[atom_k].x, fk.x);
        atomicAdd(&frc[atom_k].y, fk.y);
        atomicAdd(&frc[atom_k].z, fk.z);
        atomicAdd(&frc[atom_l].x, fl.x);
        atomicAdd(&frc[atom_l].y, fl.y);
        atomicAdd(&frc[atom_l].z, fl.z);

        atomicAdd(&ene[atom_i], (temp_pk + cos_nphi * temp_gamc + sin_nphi * temp_gams));
    }
}
void DIHEDRAL::Initial(CONTROLLER *controller, const char *module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "dihedral");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }


    char file_name_suffix[CHAR_LENGTH_MAX];
    sprintf(file_name_suffix, "in_file");


    if (controller[0].Command_Exist(this->module_name, file_name_suffix))
    {
        controller[0].printf("START INITIALIZING DIHEDRAL (%s_%s):\n", this->module_name, file_name_suffix);

        FILE *fp = NULL;
        Open_File_Safely(&fp, controller[0].Command(this->module_name, file_name_suffix), "r");

        int scanf_ret = fscanf(fp, "%d", &dihedral_numbers);
        controller[0].printf("    dihedral_numbers is %d\n", dihedral_numbers);
        Memory_Allocate();

        
        float temp;
        for (int i = 0; i < dihedral_numbers; i++)
        {
            scanf_ret = fscanf(fp, "%d %d %d %d %d %f %f", h_atom_a + i, h_atom_b + i, h_atom_c + i, h_atom_d + i, h_ipn + i, h_pk + i, &temp);
            h_pn[i] = (float) h_ipn[i];
            h_gamc[i] = cosf(temp) * h_pk[i];
            h_gams[i] = sinf(temp )* h_pk[i];
        }
        fclose(fp);
        Parameter_Host_To_Device();
        is_initialized = 1;
    }
    else if (controller[0].Command_Exist("amber_parm7"))
    {
        controller[0].printf("START INITIALIZING DIHEDRAL (amber_parm7):\n");
        Read_Information_From_AMBERFILE(controller[0].Command("amber_parm7"), controller[0]);
        if (dihedral_numbers > 0)
            is_initialized = 1;
    }
    else
    {
        controller[0].printf("DIHEDRAL IS NOT INITIALIZED\n\n");
    }

    if (is_initialized && !is_controller_printf_initialized)
    {
        controller[0].Step_Print_Initial(this->module_name, "%.2f");
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }
    if (is_initialized)
    {
        controller[0].printf("END INITIALIZING DIHEDRAL\n\n");
    }
}
void DIHEDRAL::Read_Information_From_AMBERFILE(const char *file_name, CONTROLLER controller)
{
    float *phase_type_cpu = NULL, *pk_type_cpu = NULL, *pn_type_cpu = NULL;
    int dihedral_type_numbers = 0, dihedral_with_hydrogen = 0;
    FILE *parm = NULL;
    Open_File_Safely(&parm, file_name, "r");
    char temps[CHAR_LENGTH_MAX];
    char temp_first_str[CHAR_LENGTH_MAX];
    char temp_second_str[CHAR_LENGTH_MAX];
    int i, tempi;
    float tempf, tempf2;
    controller.printf("    Reading dihedral information from AMBER file:\n");
    while (true)
    {
        if (!fgets(temps, CHAR_LENGTH_MAX, parm))
            break;
        if (sscanf(temps, "%s %s", temp_first_str, temp_second_str) != 2)
        {
            continue;
        }
        if (strcmp(temp_first_str, "%FLAG") == 0
            && strcmp(temp_second_str, "POINTERS") == 0)
        {
            char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);

            for (i = 0; i < 6; i++)
                int scanf_ret = fscanf(parm, "%d", &tempi);

            int scanf_ret = fscanf(parm, "%d", &dihedral_with_hydrogen);
            scanf_ret = fscanf(parm, "%d", &this->dihedral_numbers);
            this->dihedral_numbers += dihedral_with_hydrogen;
            
            controller.printf("        dihedral numbers is %d\n", this->dihedral_numbers);

            this->Memory_Allocate();

            for (i = 0; i < 9; i++)
                scanf_ret = fscanf(parm, "%d", &tempi);

            scanf_ret = fscanf(parm, "%d", &dihedral_type_numbers);
            controller.printf("        dihedral type numbers is %d\n", dihedral_type_numbers);

            Malloc_Safely((void**)&phase_type_cpu, sizeof(float)* dihedral_type_numbers);
            Malloc_Safely((void**)&pk_type_cpu, sizeof(float)* dihedral_type_numbers);
            Malloc_Safely((void**)&pn_type_cpu, sizeof(float)* dihedral_type_numbers);
        }
        if (strcmp(temp_first_str, "%FLAG") == 0
            && strcmp(temp_second_str, "DIHEDRAL_FORCE_CONSTANT") == 0)
        {
            controller.printf("        read dihedral force constant\n");
            char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            for (i = 0; i < dihedral_type_numbers; i++)
                int scanf_ret = fscanf(parm, "%f", &pk_type_cpu[i]);
        }

        if (strcmp(temp_first_str, "%FLAG") == 0
            && strcmp(temp_second_str, "DIHEDRAL_PHASE") == 0)
        {
            controller.printf("        read dihedral phase\n");
            char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            for (i = 0; i<dihedral_type_numbers; i++)
                int scanf_ret = fscanf(parm, "%f", &phase_type_cpu[i]);
        }

        if (strcmp(temp_first_str, "%FLAG") == 0
            && strcmp(temp_second_str, "DIHEDRAL_PERIODICITY") == 0)
        {
            controller.printf("        read dihedral periodicity\n");
            char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            for (i = 0; i<dihedral_type_numbers; i++)
                int scanf_ret = fscanf(parm, "%f", &pn_type_cpu[i]);
        }

        if (strcmp(temp_first_str, "%FLAG") == 0
            && strcmp(temp_second_str, "DIHEDRALS_INC_HYDROGEN") == 0)
        {
            char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            for (i = 0; i < dihedral_with_hydrogen; i++)
            {
                int scanf_ret = fscanf(parm, "%d\n", &this->h_atom_a[i]);
                scanf_ret = fscanf(parm, "%d\n", &this->h_atom_b[i]);
                scanf_ret = fscanf(parm, "%d\n", &this->h_atom_c[i]);
                scanf_ret = fscanf(parm, "%d\n", &this->h_atom_d[i]);
                scanf_ret = fscanf(parm, "%d\n", &tempi);
                this->h_atom_a[i] /= 3;
                this->h_atom_b[i] /= 3;
                this->h_atom_c[i] /= 3;

                this->h_atom_d[i] = abs(this->h_atom_d[i] / 3);
                tempi -= 1;
                this->h_pk[i] = pk_type_cpu[tempi];

                tempf = phase_type_cpu[tempi];
                if (abs(tempf - CONSTANT_Pi) <= 0.001)
                    tempf = CONSTANT_Pi;

                tempf2 = cosf(tempf);
                if (fabsf(tempf2) < 1e-6)
                    tempf2 = 0;
                this->h_gamc[i] = tempf2 * this->h_pk[i];

                tempf2 = sinf(tempf);
                if (fabsf(tempf2) < 1e-6)
                    tempf2 = 0;
                this->h_gams[i] = tempf2 * this->h_pk[i];

                this->h_pn[i] = fabsf(pn_type_cpu[tempi]);
                this->h_ipn[i] = (int)(this->h_pn[i] + 0.001);
                
            }
        }
        if (strcmp(temp_first_str, "%FLAG") == 0
            && strcmp(temp_second_str, "DIHEDRALS_WITHOUT_HYDROGEN") == 0)
        {
            char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
            for (i = dihedral_with_hydrogen; i < this->dihedral_numbers; i++)
            {
                int scanf_ret = fscanf(parm, "%d\n", &this->h_atom_a[i]);
                scanf_ret = fscanf(parm, "%d\n", &this->h_atom_b[i]);
                scanf_ret = fscanf(parm, "%d\n", &this->h_atom_c[i]);
                scanf_ret = fscanf(parm, "%d\n", &this->h_atom_d[i]);
                scanf_ret = fscanf(parm, "%d\n", &tempi);
                this->h_atom_a[i] /= 3;
                this->h_atom_b[i] /= 3;
                this->h_atom_c[i] /= 3;

                this->h_atom_d[i] = abs(this->h_atom_d[i] / 3);
                tempi -= 1;
                this->h_pk[i] = pk_type_cpu[tempi];

                tempf = phase_type_cpu[tempi];
                if (abs(tempf - CONSTANT_Pi) <= 0.001)
                    tempf = CONSTANT_Pi;

                tempf2 = cosf(tempf);
                if (fabsf(tempf2) < 1e-6)
                    tempf2 = 0;
                this->h_gamc[i] = tempf2 * this->h_pk[i];

                tempf2 = sinf(tempf);
                if (fabsf(tempf2) < 1e-6)
                    tempf2 = 0;
                this->h_gams[i] = tempf2 * this->h_pk[i];

                this->h_pn[i] = fabsf(pn_type_cpu[tempi]);
                this->h_ipn[i] = (int)(this->h_pn[i] + 0.001);
        
            }
        }
    }
    
    for (int i = 0; i < this->dihedral_numbers; ++i)
    {
        if (this->h_atom_c[i] < 0)
            this->h_atom_c[i] *= -1;
    }
    
    controller.printf("    End reading dihedral information from AMBER file\n");
    
    fclose(parm);
    free(pn_type_cpu);
    free(phase_type_cpu);
    free(pk_type_cpu);
    
    Parameter_Host_To_Device();
    is_initialized = 1;
    if (dihedral_numbers == 0)
        Clear();
}

void DIHEDRAL::Memory_Allocate()
{
    if (!Malloc_Safely((void**)&this->h_atom_a, sizeof(int)* this->dihedral_numbers))
        printf("Error occurs when malloc DIHEDARL::h_atom_a in DIHEDRAL::Dihedral_Initialize");
    if (!Malloc_Safely((void**)&this->h_atom_b, sizeof(int)* this->dihedral_numbers))
        printf("Error occurs when malloc DIHEDARL::h_atom_b in DIHEDRAL::Dihedral_Initialize");
    if (!Malloc_Safely((void**)&this->h_atom_c, sizeof(int)* this->dihedral_numbers))
        printf("Error occurs when malloc DIHEDARL::h_atom_c in DIHEDRAL::Dihedral_Initialize");
    if (!Malloc_Safely((void**)&this->h_atom_d, sizeof(int)* this->dihedral_numbers))
        printf("Error occurs when malloc DIHEDARL::h_atom_d in DIHEDRAL::Dihedral_Initialize");
    if (!Malloc_Safely((void**)&this->h_ipn, sizeof(int)* this->dihedral_numbers))
        printf("Error occurs when malloc DIHEDARL::h_ipn in DIHEDRAL::Dihedral_Initialize");
    if (!Malloc_Safely((void**)&this->h_pk, sizeof(float)* this->dihedral_numbers))
        printf("Error occurs when malloc DIHEDARL::h_pk in DIHEDRAL::Dihedral_Initialize");
    if (!Malloc_Safely((void**)&this->h_gamc, sizeof(float)* this->dihedral_numbers))
        printf("Error occurs when malloc DIHEDARL::h_gamc in DIHEDRAL::Dihedral_Initialize");
    if (!Malloc_Safely((void**)&this->h_gams, sizeof(float)* this->dihedral_numbers))
        printf("Error occurs when malloc DIHEDARL::h_gams in DIHEDRAL::Dihedral_Initialize");
    if (!Malloc_Safely((void**)&this->h_pn, sizeof(float)* this->dihedral_numbers))
        printf("Error occurs when malloc DIHEDARL::h_pn in DIHEDRAL::Dihedral_Initialize");
    if (!Malloc_Safely((void**)&this->h_dihedral_ene, sizeof(float)* this->dihedral_numbers))
        printf("Error occurs when malloc DIHEDARL::h_dihedral_energy in DIHEDRAL::Dihedral_Initialize");
    if (!Malloc_Safely((void**)&this->h_sigma_of_dihedral_ene, sizeof(float)))
        printf("Error occurs when malloc DIHEDARL::h_sigma_energy in DIHEDRAL::Dihedral_Initialize");


    //CUDA
    if (!Cuda_Malloc_Safely((void**)&this->d_atom_a, sizeof(int)* this->dihedral_numbers))
        printf("Error occurs when CUDA malloc DIHEDARL::d_atom_a in DIHEDRAL::Dihedral_Initialize");
    if (!Cuda_Malloc_Safely((void**)&this->d_atom_b, sizeof(int)* this->dihedral_numbers))
        printf("Error occurs when CUDA malloc DIHEDARL::d_atom_b in DIHEDRAL::Dihedral_Initialize");
    if (!Cuda_Malloc_Safely((void**)&this->d_atom_c, sizeof(int)* this->dihedral_numbers))
        printf("Error occurs when CUDA malloc DIHEDARL::d_atom_c in DIHEDRAL::Dihedral_Initialize");
    if (!Cuda_Malloc_Safely((void**)&this->d_atom_d, sizeof(int)* this->dihedral_numbers))
        printf("Error occurs when CUDA malloc DIHEDARL::d_atom_d in DIHEDRAL::Dihedral_Initialize");
    if (!Cuda_Malloc_Safely((void**)&this->d_ipn, sizeof(int)* this->dihedral_numbers))
        printf("Error occurs when CUDA malloc DIHEDARL::d_ipn in DIHEDRAL::Dihedral_Initialize");

    if (!Cuda_Malloc_Safely((void**)&this->d_pk, sizeof(float)* this->dihedral_numbers))
        printf("Error occurs when CUDA malloc DIHEDARL::d_pk in DIHEDRAL::Dihedral_Initialize");
    if (!Cuda_Malloc_Safely((void**)&this->d_gamc, sizeof(float)* this->dihedral_numbers))
        printf("Error occurs when CUDA malloc DIHEDARL::d_gamc in DIHEDRAL::Dihedral_Initialize");
    if (!Cuda_Malloc_Safely((void**)&this->d_gams, sizeof(float)* this->dihedral_numbers))
        printf("Error occurs when CUDA malloc DIHEDARL::d_gams in DIHEDRAL::Dihedral_Initialize");
    if (!Cuda_Malloc_Safely((void**)&this->d_pn, sizeof(float)* this->dihedral_numbers))
        printf("Error occurs when CUDA malloc DIHEDARL::d_pn in DIHEDRAL::Dihedral_Initialize");
    if (!Cuda_Malloc_Safely((void**)&this->d_dihedral_ene, sizeof(float)* this->dihedral_numbers))
        printf("Error occurs when CUDA malloc DIHEDARL::d_dihedral_energy in DIHEDRAL::Dihedral_Initialize");
    if (!Cuda_Malloc_Safely((void**)&this->d_sigma_of_dihedral_ene, sizeof(float)))
        printf("Error occurs when CUDA malloc DIHEDARL::d_sigma_energy in DIHEDRAL::Dihedral_Initialize");
}

void DIHEDRAL::Parameter_Host_To_Device()
{
    cudaMemcpy(this->d_atom_a, this->h_atom_a, sizeof(int)*this->dihedral_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_atom_b, this->h_atom_b, sizeof(int)*this->dihedral_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_atom_c, this->h_atom_c, sizeof(int)*this->dihedral_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_atom_d, this->h_atom_d, sizeof(int)*this->dihedral_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_ipn, this->h_ipn, sizeof(int)*this->dihedral_numbers, cudaMemcpyHostToDevice);

    cudaMemcpy(this->d_pk, this->h_pk, sizeof(float)*this->dihedral_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_pn, this->h_pn, sizeof(float)*this->dihedral_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_gamc, this->h_gamc, sizeof(float)*this->dihedral_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_gams, this->h_gams, sizeof(float)*this->dihedral_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_dihedral_ene, this->h_dihedral_ene, sizeof(float)*this->dihedral_numbers, cudaMemcpyHostToDevice);

    cudaMemcpy(this->d_sigma_of_dihedral_ene, this->h_sigma_of_dihedral_ene, sizeof(float), cudaMemcpyHostToDevice);
}

void DIHEDRAL::Clear()
{
    if (is_initialized)
    {
        is_initialized = 0;

        free(this->h_atom_a);
        cudaFree(this->d_atom_a);
        free(this->h_atom_b);
        cudaFree(this->d_atom_b);
        free(this->h_atom_c);
        cudaFree(this->d_atom_c);
        free(this->h_atom_d);
        cudaFree(this->d_atom_d);
    
        free(this->h_pk);
        cudaFree(this->d_pk);
        free(this->h_pn);
        cudaFree(this->d_pn);
        free(this->h_ipn);
        cudaFree(this->d_ipn);
        free(this->h_gamc);
        cudaFree(this->d_gamc);
        free(this->h_gams);
        cudaFree(this->d_gams);

        free(this->h_dihedral_ene);
        cudaFree(this->d_dihedral_ene);
        free(this->h_sigma_of_dihedral_ene);
        cudaFree(this->d_sigma_of_dihedral_ene);

        this->h_atom_a= NULL;
        this->d_atom_a= NULL;
        this->h_atom_b= NULL;
        this->d_atom_b= NULL;
        this->h_atom_c= NULL;
        this->d_atom_c= NULL;
        this->h_atom_d= NULL;
        this->d_atom_d= NULL;

        this->h_pk= NULL;
        this->d_pk= NULL;
        this->h_pn= NULL;
        this->d_pn= NULL;
        this->h_ipn= NULL;
        this->d_ipn= NULL;
        this->h_gamc= NULL;
        this->d_gamc= NULL;
        this->h_gams= NULL;
        this->d_gams= NULL;

        this->h_dihedral_ene= NULL;
        this->d_dihedral_ene= NULL;
        this->h_sigma_of_dihedral_ene= NULL;
        this->d_sigma_of_dihedral_ene= NULL;
    }
}


float DIHEDRAL::Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, int is_download)
{
    if (is_initialized)
    {
        Dihedral_Energy_CUDA << <(unsigned int)ceilf((float)this->dihedral_numbers / this->threads_per_block), this->threads_per_block >> >
            (this->dihedral_numbers, uint_crd, scaler,
            this->d_atom_a, this->d_atom_b, this->d_atom_c, this->d_atom_d, this->d_ipn, this->d_pk, this->d_gamc,
            this->d_gams, this->d_pn, this->d_dihedral_ene);
        Sum_Of_List << <1, 1024 >> >(this->dihedral_numbers, this->d_dihedral_ene, this->d_sigma_of_dihedral_ene);
    
        if (is_download)
        {
            cudaMemcpy(this->h_sigma_of_dihedral_ene, this->d_sigma_of_dihedral_ene, sizeof(float), cudaMemcpyDeviceToHost);
            return h_sigma_of_dihedral_ene[0];
        }
        else
        {
            return 0;
        }
    }
    return NAN;
}

  
 
void DIHEDRAL::Dihedral_Force_With_Atom_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, VECTOR *frc, float *atom_energy)
{
    if (is_initialized)
    {
        Dihedral_Force_With_Atom_Energy_CUDA << <(unsigned int)ceilf((float)this->dihedral_numbers / this->threads_per_block), this->threads_per_block >> >
            (this->dihedral_numbers, uint_crd, scaler,
            this->d_atom_a, this->d_atom_b, this->d_atom_c, this->d_atom_d, this->d_ipn, this->d_pk, this->d_gamc, this->d_gams, this->d_pn, frc,
            atom_energy);
    }
}
