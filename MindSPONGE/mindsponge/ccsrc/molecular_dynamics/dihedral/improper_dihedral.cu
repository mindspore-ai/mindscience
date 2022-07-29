#include "improper_dihedral.cuh"
 
static __global__ void Dihedral_Energy_CUDA(const int dihedral_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler,
    const int *atom_a, const int *atom_b, const int *atom_c, const int *atom_d, const float *pk, const float *phi0, float *ene)
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
        float temp_phi0 = phi0[dihedral_i];

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
                //printf("%f\n", phi / 3.1415926 * 180);
        float delta_phi = phi - temp_phi0;
                
        if (delta_phi > CONSTANT_Pi)
        {
            delta_phi -= 2.0f * CONSTANT_Pi;
        }
        else if (delta_phi < -CONSTANT_Pi)
        {
            delta_phi += 2.0f * CONSTANT_Pi;
        }
        ene[dihedral_i] = temp_pk * delta_phi * delta_phi;
    }
}

static __global__ void Dihedral_Force_With_Atom_Energy_CUDA(const int dihedral_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler,
    const int *atom_a, const int *atom_b, const int *atom_c, const int *atom_d, const float *pk, const float *phi0, VECTOR *frc,
    float *ene)
{
    int dihedral_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (dihedral_i < dihedral_numbers)
    {
        int atom_i = atom_a[dihedral_i];
        int atom_j = atom_b[dihedral_i];
        int atom_k = atom_c[dihedral_i];
        int atom_l = atom_d[dihedral_i];

        float temp_phi0 = phi0[dihedral_i];

        float temp_pk = pk[dihedral_i];

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
                
        float delta_phi = phi - temp_phi0;
        if (delta_phi > CONSTANT_Pi)
        {
            delta_phi -= 2.0f * CONSTANT_Pi;
        }
        else if (delta_phi < -CONSTANT_Pi)
        {
            delta_phi += 2.0f * CONSTANT_Pi;
        }
        
        atomicAdd(&ene[atom_i], temp_pk * delta_phi * delta_phi);

        float sin_phi = sinf(phi);
        float cos_phi = cosf(phi);
                
                //Here and following var name "phi" corespongding to the declaration of phi
                //aka, the var with the comment line "PHI, pay attention to the var NAME" 
                //The real dihedral = Pi - ArcCos(so-called "phi")
                //d(real dihedral) = 1/sin(real dihedral) * d(so-called  "phi")
        float dE_dphi = -2.0f * temp_pk *  delta_phi / sin_phi; 

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
    }
}

void IMPROPER_DIHEDRAL::Initial(CONTROLLER *controller, const char *module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "improper_dihedral");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }


    char file_name_suffix[CHAR_LENGTH_MAX];
    sprintf(file_name_suffix, "in_file");

    if (controller[0].Command_Exist(this->module_name, file_name_suffix))
    {
        controller[0].printf("START INITIALIZING IMPROPER DIHEDRAL (%s_%s):\n", this->module_name, file_name_suffix);

        FILE *fp = NULL;
        Open_File_Safely(&fp, controller[0].Command(this->module_name, file_name_suffix), "r");

        int ret = fscanf(fp, "%d", &dihedral_numbers);
        controller[0].printf("    dihedral_numbers is %d\n", dihedral_numbers);
        Memory_Allocate();

        
        for (int i = 0; i < dihedral_numbers; i++)
        {
            ret = fscanf(fp, "%d %d %d %d %f %f", h_atom_a + i, h_atom_b + i, h_atom_c + i, h_atom_d + i, h_pk + i, h_phi0 + i);
        }
        fclose(fp);
        Parameter_Host_To_Device();
        is_initialized = 1;
    }
    else
    {
        controller[0].printf("IMPROPER DIHEDRAL IS NOT INITIALIZED\n\n");
    }

    if (is_initialized && !is_controller_printf_initialized)
    {
        controller[0].Step_Print_Initial(this->module_name, "%.2f");
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }
    if (is_initialized)
    {
        controller[0].printf("END INITIALIZING IMPROPER DIHEDRAL\n\n");
    }
}


void IMPROPER_DIHEDRAL::Memory_Allocate()
{
    if (!Malloc_Safely((void**)&this->h_atom_a, sizeof(int)* this->dihedral_numbers))
        printf("Error occurs when malloc IMPROPER_DIHEDRAL::h_atom_a in IMPROPER_DIHEDRAL::Dihedral_Initialize");
    if (!Malloc_Safely((void**)&this->h_atom_b, sizeof(int)* this->dihedral_numbers))
        printf("Error occurs when malloc IMPROPER_DIHEDRAL::h_atom_b in IMPROPER_DIHEDRAL::Dihedral_Initialize");
    if (!Malloc_Safely((void**)&this->h_atom_c, sizeof(int)* this->dihedral_numbers))
        printf("Error occurs when malloc IMPROPER_DIHEDRAL::h_atom_c in IMPROPER_DIHEDRAL::Dihedral_Initialize");
    if (!Malloc_Safely((void**)&this->h_atom_d, sizeof(int)* this->dihedral_numbers))
        printf("Error occurs when malloc IMPROPER_DIHEDRAL::h_atom_d in IMPROPER_DIHEDRAL::Dihedral_Initialize");
    if (!Malloc_Safely((void**)&this->h_pk, sizeof(float)* this->dihedral_numbers))
        printf("Error occurs when malloc IMPROPER_DIHEDRAL::h_pk in IMPROPER_DIHEDRAL::Dihedral_Initialize");
    if (!Malloc_Safely((void**)&this->h_phi0, sizeof(float)* this->dihedral_numbers))
        printf("Error occurs when malloc IMPROPER_DIHEDRAL::h_phi0 in IMPROPER_DIHEDRAL::Dihedral_Initialize");
    if (!Malloc_Safely((void**)&this->h_dihedral_ene, sizeof(float)* this->dihedral_numbers))
        printf("Error occurs when malloc IMPROPER_DIHEDRAL::h_dihedral_energy in IMPROPER_DIHEDRAL::Dihedral_Initialize");
    if (!Malloc_Safely((void**)&this->h_sigma_of_dihedral_ene, sizeof(float)))
        printf("Error occurs when malloc IMPROPER_DIHEDRAL::h_sigma_energy in IMPROPER_DIHEDRAL::Dihedral_Initialize");


    //CUDA
    if (!Cuda_Malloc_Safely((void**)&this->d_atom_a, sizeof(int)* this->dihedral_numbers))
        printf("Error occurs when CUDA malloc IMPROPER_DIHEDRAL::d_atom_a in IMPROPER_DIHEDRAL::Dihedral_Initialize");
    if (!Cuda_Malloc_Safely((void**)&this->d_atom_b, sizeof(int)* this->dihedral_numbers))
        printf("Error occurs when CUDA malloc IMPROPER_DIHEDRAL::d_atom_b in IMPROPER_DIHEDRAL::Dihedral_Initialize");
    if (!Cuda_Malloc_Safely((void**)&this->d_atom_c, sizeof(int)* this->dihedral_numbers))
        printf("Error occurs when CUDA malloc IMPROPER_DIHEDRAL::d_atom_c in IMPROPER_DIHEDRAL::Dihedral_Initialize");
    if (!Cuda_Malloc_Safely((void**)&this->d_atom_d, sizeof(int)* this->dihedral_numbers))
        printf("Error occurs when CUDA malloc IMPROPER_DIHEDRAL::d_atom_d in IMPROPER_DIHEDRAL::Dihedral_Initialize");

    if (!Cuda_Malloc_Safely((void**)&this->d_pk, sizeof(float)* this->dihedral_numbers))
        printf("Error occurs when CUDA malloc IMPROPER_DIHEDRAL::d_pk in IMPROPER_DIHEDRAL::Dihedral_Initialize");
    if (!Cuda_Malloc_Safely((void**)&this->d_phi0, sizeof(float)* this->dihedral_numbers))
        printf("Error occurs when CUDA malloc IMPROPER_DIHEDRAL::d_phi0 in IMPROPER_DIHEDRAL::Dihedral_Initialize");
    if (!Cuda_Malloc_Safely((void**)&this->d_dihedral_ene, sizeof(float)* this->dihedral_numbers))
        printf("Error occurs when CUDA malloc IMPROPER_DIHEDRAL::d_dihedral_energy in IMPROPER_DIHEDRAL::Dihedral_Initialize");
    if (!Cuda_Malloc_Safely((void**)&this->d_sigma_of_dihedral_ene, sizeof(float)))
        printf("Error occurs when CUDA malloc IMPROPER_DIHEDRAL::d_sigma_energy in IMPROPER_DIHEDRAL::Dihedral_Initialize");
}

void IMPROPER_DIHEDRAL::Parameter_Host_To_Device()
{
    cudaMemcpy(this->d_atom_a, this->h_atom_a, sizeof(int)*this->dihedral_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_atom_b, this->h_atom_b, sizeof(int)*this->dihedral_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_atom_c, this->h_atom_c, sizeof(int)*this->dihedral_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_atom_d, this->h_atom_d, sizeof(int)*this->dihedral_numbers, cudaMemcpyHostToDevice);

    cudaMemcpy(this->d_pk, this->h_pk, sizeof(float)*this->dihedral_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_phi0, this->h_phi0, sizeof(float)*this->dihedral_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_dihedral_ene, this->h_dihedral_ene, sizeof(float)*this->dihedral_numbers, cudaMemcpyHostToDevice);

    cudaMemcpy(this->d_sigma_of_dihedral_ene, this->h_sigma_of_dihedral_ene, sizeof(float), cudaMemcpyHostToDevice);
}

void IMPROPER_DIHEDRAL::Clear()
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
        free(this->h_phi0);
        cudaFree(this->d_phi0);

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

        this->h_pk = NULL;
        this->d_pk = NULL;
        this->h_phi0 = NULL;
        this->d_phi0 = NULL;

        this->h_dihedral_ene= NULL;
        this->d_dihedral_ene= NULL;
        this->h_sigma_of_dihedral_ene= NULL;
        this->d_sigma_of_dihedral_ene= NULL;
    }
}


float IMPROPER_DIHEDRAL::Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, int is_download)
{
    if (is_initialized)
    {
        Dihedral_Energy_CUDA << <(unsigned int)ceilf((float)this->dihedral_numbers / this->threads_per_block), this->threads_per_block >> >
            (this->dihedral_numbers, uint_crd, scaler,
            this->d_atom_a, this->d_atom_b, this->d_atom_c, this->d_atom_d, this->d_pk, this->d_phi0, this->d_dihedral_ene);
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



void IMPROPER_DIHEDRAL::Dihedral_Force_With_Atom_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, VECTOR *frc, float *atom_energy)
{
    if (is_initialized)
    {
        Dihedral_Force_With_Atom_Energy_CUDA << <(unsigned int)ceilf((float)this->dihedral_numbers / this->threads_per_block), this->threads_per_block >> >
            (this->dihedral_numbers, uint_crd, scaler,
            this->d_atom_a, this->d_atom_b, this->d_atom_c, this->d_atom_d, this->d_pk, this->d_phi0, frc,
            atom_energy);
    }
}
