#include "virtual_atoms.cuh"

__global__ void v0_Coordinate_Refresh(const int virtual_numbers, const VIRTUAL_TYPE_0 *v_info, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, VECTOR *coordinate)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < virtual_numbers)
    {
        VIRTUAL_TYPE_0 v_temp = v_info[i];
        int atom_v = v_temp.virtual_atom;
        int atom_1 = v_temp.from_1;
        float h = v_temp.h_double;
        VECTOR temp = coordinate[atom_1];
        temp.z = 2 * h - temp.z;
        coordinate[atom_v] = temp;
    }
}

__global__ void v1_Coordinate_Refresh(const int virtual_numbers, const VIRTUAL_TYPE_1 *v_info, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, VECTOR *coordinate)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < virtual_numbers)
    {
        VIRTUAL_TYPE_1 v_temp = v_info[i];
        int atom_v = v_temp.virtual_atom;
        int atom_1 = v_temp.from_1;
        int atom_2 = v_temp.from_2;
        float a = v_temp.a;
        VECTOR rv1 = a * Get_Periodic_Displacement(uint_crd[atom_2], uint_crd[atom_1], scaler);
        coordinate[atom_v] = coordinate[atom_1] + rv1;
    }
}

__global__ void v2_Coordinate_Refresh(const int virtual_numbers, const VIRTUAL_TYPE_2 *v_info, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, VECTOR *coordinate)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < virtual_numbers)
    {
        VIRTUAL_TYPE_2 v_temp = v_info[i];
        int atom_v = v_temp.virtual_atom;
        int atom_1 = v_temp.from_1;
        int atom_2 = v_temp.from_2;
        int atom_3 = v_temp.from_3;
        float a = v_temp.a;
        float b = v_temp.b;
        
        UNSIGNED_INT_VECTOR uint_r1 = uint_crd[atom_1];
        UNSIGNED_INT_VECTOR uint_r2 = uint_crd[atom_2];
        UNSIGNED_INT_VECTOR uint_r3 = uint_crd[atom_3];

        VECTOR rv1 = a * Get_Periodic_Displacement(uint_r2, uint_r1, scaler)
            + b * Get_Periodic_Displacement(uint_r3, uint_r1, scaler);


        coordinate[atom_v] = coordinate[atom_1] + rv1;
    }
}

__global__ void v3_Coordinate_Refresh(const int virtual_numbers, const VIRTUAL_TYPE_3 *v_info, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, VECTOR *coordinate)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < virtual_numbers)
    {
        VIRTUAL_TYPE_3 v_temp = v_info[i];
        int atom_v = v_temp.virtual_atom;
        int atom_1 = v_temp.from_1;
        int atom_2 = v_temp.from_2;
        int atom_3 = v_temp.from_3;
        float d = v_temp.d;
        float k = v_temp.k;
        UNSIGNED_INT_VECTOR uint_r1 = uint_crd[atom_1];
        UNSIGNED_INT_VECTOR uint_r2 = uint_crd[atom_2];
        UNSIGNED_INT_VECTOR uint_r3 = uint_crd[atom_3];

        VECTOR r21 = Get_Periodic_Displacement(uint_r2, uint_r1, scaler);
        VECTOR r32 = Get_Periodic_Displacement(uint_r3, uint_r2, scaler);

        VECTOR temp = r21 + k * r32;
        temp = d * rnorm3df(temp.x, temp.y, temp.z) * temp;
        coordinate[atom_v] =  coordinate[atom_1] + temp;
    }
}

__global__ void v0_Force_Redistribute(const int virtual_numbers, const VIRTUAL_TYPE_0 *v_info, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler,  VECTOR *force)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < virtual_numbers)
    {
        VIRTUAL_TYPE_0 v_temp = v_info[i];
        int atom_v = v_temp.virtual_atom;
        int atom_1 = v_temp.from_1;
        VECTOR force_v = force[atom_v];
        atomicAdd(&force[atom_1].x, force_v.x);
        atomicAdd(&force[atom_1].y, force_v.y);
        atomicAdd(&force[atom_1].z, -force_v.z);
        force_v.x = 0.0f;
        force_v.y = 0.0f;
        force_v.z = 0.0f;
        force[atom_v] = force_v;
    }
}

__global__ void v1_Force_Redistribute(const int virtual_numbers, const VIRTUAL_TYPE_1 *v_info, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, VECTOR *force)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < virtual_numbers)
    {
        VIRTUAL_TYPE_1 v_temp = v_info[i];
        int atom_v = v_temp.virtual_atom;
        int atom_1 = v_temp.from_1;
        int atom_2 = v_temp.from_2;
        float a = v_temp.a;
        VECTOR force_v = force[atom_v];
        atomicAdd(&force[atom_1].x, a * force_v.x);
        atomicAdd(&force[atom_1].y, a * force_v.y);
        atomicAdd(&force[atom_1].z, a * force_v.z);
        
        atomicAdd(&force[atom_2].x, (1 - a) * force_v.x);
        atomicAdd(&force[atom_2].y, (1 - a) * force_v.y);
        atomicAdd(&force[atom_2].z, (1 - a) * force_v.z);

        force_v.x = 0.0f;
        force_v.y = 0.0f;
        force_v.z = 0.0f;
        force[atom_v] = force_v;
    }
}

__global__ void v2_Force_Redistribute(const int virtual_numbers, const VIRTUAL_TYPE_2 *v_info, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, VECTOR *force)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < virtual_numbers)
    {
        VIRTUAL_TYPE_2 v_temp = v_info[i];
        int atom_v = v_temp.virtual_atom;
        int atom_1 = v_temp.from_1;
        int atom_2 = v_temp.from_2;
        int atom_3 = v_temp.from_3;
        float a = v_temp.a;
        float b = v_temp.b;
        VECTOR force_v = force[atom_v];
        atomicAdd(&force[atom_1].x, (1 - a - b) * force_v.x);
        atomicAdd(&force[atom_1].y, (1 - a - b) * force_v.y);
        atomicAdd(&force[atom_1].z, (1 - a - b) * force_v.z);
        
        atomicAdd(&force[atom_2].x, a * force_v.x);
        atomicAdd(&force[atom_2].y, a * force_v.y);
        atomicAdd(&force[atom_2].z, a * force_v.z);
        
        atomicAdd(&force[atom_3].x, b * force_v.x);
        atomicAdd(&force[atom_3].y, b * force_v.y);
        atomicAdd(&force[atom_3].z, b * force_v.z);

        force_v.x = 0.0f;
        force_v.y = 0.0f;
        force_v.z = 0.0f;
        force[atom_v] = force_v;
    }
}

__global__ void v3_Force_Redistribute(const int virtual_numbers, const VIRTUAL_TYPE_3 *v_info, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, VECTOR *force)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < virtual_numbers)
    {
        VIRTUAL_TYPE_3 v_temp = v_info[i];
        int atom_v = v_temp.virtual_atom;
        int atom_1 = v_temp.from_1;
        int atom_2 = v_temp.from_2;
        int atom_3 = v_temp.from_3;
        float d = v_temp.d;
        float k = v_temp.k;
        VECTOR force_v = force[atom_v];

        UNSIGNED_INT_VECTOR uint_r1 = uint_crd[atom_1];
        UNSIGNED_INT_VECTOR uint_r2 = uint_crd[atom_2];
        UNSIGNED_INT_VECTOR uint_r3 = uint_crd[atom_3];
        UNSIGNED_INT_VECTOR uint_rv = uint_crd[atom_v];

        VECTOR r21 = Get_Periodic_Displacement(uint_r2, uint_r1, scaler);
        VECTOR r32 = Get_Periodic_Displacement(uint_r3, uint_r2, scaler);
        VECTOR rv1 = Get_Periodic_Displacement(uint_rv, uint_r1, scaler);

        VECTOR temp = r21 + k * r32;
        float factor = d * rnorm3df(temp.x, temp.y, temp.z);

        temp = (rv1 * force_v) / (rv1 * rv1) * rv1;
        temp = factor * (force_v - temp);
        VECTOR force_1 = force_v - temp;
        VECTOR force_2 = (1 - k) * temp;
        VECTOR force_3 = k * temp;
        
        atomicAdd(&force[atom_1].x, force_1.x);
        atomicAdd(&force[atom_1].y, force_1.y);
        atomicAdd(&force[atom_1].z, force_1.z);
        
        atomicAdd(&force[atom_2].x, force_2.x);
        atomicAdd(&force[atom_2].y, force_2.y);
        atomicAdd(&force[atom_2].z, force_2.z);
        
        atomicAdd(&force[atom_3].x, force_3.x);
        atomicAdd(&force[atom_3].y, force_3.y);
        atomicAdd(&force[atom_3].z, force_3.z);

        force_v.x = 0.0f;
        force_v.y = 0.0f;
        force_v.z = 0.0f;
        force[atom_v] = force_v;
    }
}


void VIRTUAL_INFORMATION::Initial( CONTROLLER *controller, int atom_numbers, int *system_freedom, const char *module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "virtual_atom");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (controller[0].Command_Exist("virtual_atom_in_file"))
    {
        printf("START INITIALIZING VIRTUAL ATOM\n");
        FILE *fp = NULL;
        char line[CHAR_LENGTH_MAX];
        Open_File_Safely(&fp, controller[0].Command("virtual_atom_in_file"), "r");
        
        Malloc_Safely((void**)&virtual_level, sizeof(int) * atom_numbers);
        for (int i = 0; i < atom_numbers; i++)
        {
            virtual_level[i] = 0;
        }
        
        int virtual_type;
        int virtual_atom;
        int temp[12];
        float tempf[5];
        int scanf_ret;

        
        //文件会从头到尾读三遍，分别确定每个原子的虚拟等级（因为可能存在坐标依赖于虚原子的虚原子，所以不得不如此做）
        //第一遍确定虚拟原子的层级
        controller[0].printf("    Start reading virtual levels\n");
        int line_numbers = 0;
        while (fgets(line, 1024, fp) != NULL)
        {
            line_numbers++;
            scanf_ret = sscanf(line, "%d %d", &virtual_type, &virtual_atom);
            if (scanf_ret != 2)
            {
                continue;
            }
            switch (virtual_type)
            {
                case 0:
                    scanf_ret = sscanf(line, "%*d %*d %d %f", temp, tempf);
                    if (scanf_ret != 2)
                    {
                        controller[0].printf("        Error: can not parse line #%d.\n", line_numbers);
                        getchar();
                        exit(1);
                    }
                    virtual_level[virtual_atom] = virtual_level[temp[0]] + 1;
                    break;

                case 1:
                    scanf_ret = sscanf(line, "%*d %*d %d %d %f", temp, temp + 1, tempf);
                    if (scanf_ret != 3)
                    {
                        controller[0].printf("        Error: can not parse line #%d.\n", line_numbers);
                        getchar();
                        exit(1);
                    }
                    virtual_level[virtual_atom] = std::max(virtual_level[temp[0]], virtual_level[temp[1]])+ 1;
                    break;

                case 2:
                    scanf_ret = sscanf(line, "%*d %*d %d %d %d %f %f", temp, temp + 1, temp + 2, tempf, tempf + 1);
                    if (scanf_ret != 5)
                    {
                        controller[0].printf("        Error: can not parse line #%d.\n", line_numbers);
                        getchar();
                        exit(1);
                    }
                    virtual_level[virtual_atom] = std::max(virtual_level[temp[0]], virtual_level[temp[1]]);
                    virtual_level[virtual_atom] = std::max(virtual_level[virtual_atom], virtual_level[temp[2]]) + 1;
                    break;

                case 3:
                    scanf_ret = sscanf(line, "%*d %*d %d %d %d %f %f", temp, temp + 1, temp + 2, tempf, tempf + 1);
                    if (scanf_ret != 5)
                    {
                        controller[0].printf("        Error: can not parse line #%d.\n", line_numbers);
                        getchar();
                        exit(1);
                    }
                    virtual_level[virtual_atom] = std::max(virtual_level[temp[0]], virtual_level[temp[1]]);
                    virtual_level[virtual_atom] = std::max(virtual_level[virtual_atom], virtual_level[temp[2]]) + 1;
                    break;

                default:
                    controller[0].printf("        Error: can not parse line #%d because %d is not a proper type for virtual atoms.\n", line_numbers, virtual_type);
                    getchar();
                    exit(1);
            }
        }
        
        //层级初始化
        max_level = 0;
        int total_virtual_atoms = 0;
        for (int i = 0; i < atom_numbers; i++)
        {
            int vli = virtual_level[i];
            if (vli > 0)
            {
                total_virtual_atoms++;
            }
            if (vli > max_level)
            {
                for (int j = 0; j < vli - max_level; j++)
                {
                    VIRTUAL_LAYER_INFORMATION virtual_layer;
                    virtual_layer_info.push_back(virtual_layer);
                }
                max_level = vli;
            }
        }
        system_freedom[0] -= 3 * total_virtual_atoms;
        controller[0].printf("        Virtual Atoms Max Level is %d\n", max_level);
        controller[0].printf("        Virtual Atoms Number is %d\n", total_virtual_atoms);
        controller[0].printf("    End reading virtual levels\n");
        //第二遍确定虚拟原子每一层的个数
        controller[0].printf("    Start reading virtual type numbers in different levels\n");
        fseek(fp, 0, SEEK_SET);
        line_numbers = 0;
        while (fgets(line, 1024, fp) != NULL)
        {
            line_numbers++;
            scanf_ret = sscanf(line, "%d %d", &virtual_type, &virtual_atom);
            if (scanf_ret != 2)
            {
                continue;
            }
            VIRTUAL_LAYER_INFORMATION *temp_vl = &virtual_layer_info[virtual_level[virtual_atom] - 1];
            switch (virtual_type)
            {
                case 0:
                    temp_vl[0].v0_info.virtual_numbers += 1;
                    break;
                case 1:
                    temp_vl[0].v1_info.virtual_numbers += 1;
                    break;
                case 2:
                    temp_vl[0].v2_info.virtual_numbers += 1;
                    break;
                case 3:
                    temp_vl[0].v3_info.virtual_numbers += 1;
                    break;
                default:
                    break;
            }
        }
        
        //每层的每种虚拟原子初始化
        for (int layer = 0; layer < max_level; layer++)
        {
            controller[0].printf("        Virutual level %d:\n", layer);
            VIRTUAL_LAYER_INFORMATION *temp_vl = &virtual_layer_info[layer];
            if (temp_vl[0].v0_info.virtual_numbers > 0)
            {
                controller[0].printf("            Virtual type 0 atom numbers is %d\n", temp_vl[0].v0_info.virtual_numbers);
                Malloc_Safely((void**)&temp_vl[0].v0_info.h_virtual_type_0, sizeof(VIRTUAL_TYPE_0) * temp_vl[0].v0_info.virtual_numbers);
                Cuda_Malloc_Safely((void**)&temp_vl[0].v0_info.d_virtual_type_0, sizeof(VIRTUAL_TYPE_0) * temp_vl[0].v0_info.virtual_numbers);
            }
            if (temp_vl[0].v1_info.virtual_numbers > 0)
            {
                controller[0].printf("            Virtual type 1 atom numbers is %d\n", temp_vl[0].v1_info.virtual_numbers);
                Malloc_Safely((void**)&temp_vl[0].v1_info.h_virtual_type_1, sizeof(VIRTUAL_TYPE_1)* temp_vl[0].v1_info.virtual_numbers);
                Cuda_Malloc_Safely((void**)&temp_vl[0].v1_info.d_virtual_type_1, sizeof(VIRTUAL_TYPE_1)* temp_vl[0].v1_info.virtual_numbers);
            }
            if (temp_vl[0].v2_info.virtual_numbers > 0)
            {
                controller[0].printf("            Virtual type 2 atom numbers is %d\n", temp_vl[0].v2_info.virtual_numbers);
                Malloc_Safely((void**)&temp_vl[0].v2_info.h_virtual_type_2, sizeof(VIRTUAL_TYPE_2) * temp_vl[0].v2_info.virtual_numbers);
                Cuda_Malloc_Safely((void**)&temp_vl[0].v2_info.d_virtual_type_2, sizeof(VIRTUAL_TYPE_2) * temp_vl[0].v2_info.virtual_numbers);
            }
            if (temp_vl[0].v3_info.virtual_numbers > 0)
            {
                controller[0].printf("            Virtual type 3 atom numbers is %d\n", temp_vl[0].v3_info.virtual_numbers);
                Malloc_Safely((void**)&temp_vl[0].v3_info.h_virtual_type_3, sizeof(VIRTUAL_TYPE_3)* temp_vl[0].v3_info.virtual_numbers);
                Cuda_Malloc_Safely((void**)&temp_vl[0].v3_info.d_virtual_type_3, sizeof(VIRTUAL_TYPE_3)* temp_vl[0].v3_info.virtual_numbers);
            }
        }
        controller[0].printf("    End reading virtual type numbers in different levels\n");
        //第三遍将所有信息填入
        controller[0].printf("    Start reading information for every virtual atom\n");
        fseek(fp, 0, SEEK_SET);
        line_numbers = 0;
        int count0 = 0, count1 = 0, count2 = 0, count3 = 0;
        while (fgets(line, 1024, fp) != NULL)
        {
            line_numbers++;
            scanf_ret = sscanf(line, "%d %d", &virtual_type, &virtual_atom);
            if (scanf_ret != 2)
            {
                continue;
            }
            VIRTUAL_LAYER_INFORMATION *temp_vl = &virtual_layer_info[virtual_level[virtual_atom] - 1];
            switch (virtual_type)
            {
                case 0:
                    scanf_ret = sscanf(line, "%*d %d %d %f", &temp_vl[0].v0_info.h_virtual_type_0[count0].virtual_atom, 
                    &temp_vl[0].v0_info.h_virtual_type_0[count0].from_1, &temp_vl[0].v0_info.h_virtual_type_0[count0].h_double);
                    temp_vl[0].v0_info.h_virtual_type_0[count0].h_double *= 2;
                    if (scanf_ret != 3)
                    {
                        continue;
                    }
                    count0++;
                    break;

                case 1:
                    scanf_ret = sscanf(line, "%*d %d %d %d %f", &temp_vl[0].v1_info.h_virtual_type_1[count0].virtual_atom,
                    &temp_vl[0].v1_info.h_virtual_type_1[count1].from_1, &temp_vl[0].v1_info.h_virtual_type_1[count1].from_2,
                    &temp_vl[0].v1_info.h_virtual_type_1[count1].a);
                    if (scanf_ret != 4)
                    {
                        continue;
                    }
                    count1++;
                    break;

                case 2:
                    scanf_ret = sscanf(line, "%*d %d %d %d %d %f %f", &temp_vl[0].v2_info.h_virtual_type_2[count2].virtual_atom,
                    &temp_vl[0].v2_info.h_virtual_type_2[count2].from_1, &temp_vl[0].v2_info.h_virtual_type_2[count2].from_2,
                    &temp_vl[0].v2_info.h_virtual_type_2[count2].from_3, &temp_vl[0].v2_info.h_virtual_type_2[count2].a, 
                    &temp_vl[0].v2_info.h_virtual_type_2[count2].b);
                    if (scanf_ret != 6)
                    {
                        continue;
                    }
                    count2++;
                    break;

                case 3:
                    scanf_ret = sscanf(line, "%*d %d %d %d %d %f %f", &temp_vl[0].v3_info.h_virtual_type_3[count3].virtual_atom,
                    &temp_vl[0].v3_info.h_virtual_type_3[count3].from_1, &temp_vl[0].v3_info.h_virtual_type_3[count3].from_2,
                    &temp_vl[0].v3_info.h_virtual_type_3[count3].from_3, &temp_vl[0].v3_info.h_virtual_type_3[count3].d, 
                    &temp_vl[0].v3_info.h_virtual_type_3[count3].k);
                    if (scanf_ret != 6)
                    {
                        continue;
                    }
                    count3++;
                    break;

                default:
                    break;
            }
        }
        fclose(fp);
        
        //每层的数据信息传到cuda上去
        for (int layer = 0; layer < max_level; layer++)
        {
            VIRTUAL_LAYER_INFORMATION temp_vl = virtual_layer_info[layer];
            if (temp_vl.v0_info.virtual_numbers>0)
                cudaMemcpy(temp_vl.v0_info.d_virtual_type_0, temp_vl.v0_info.h_virtual_type_0, sizeof(VIRTUAL_TYPE_0) * temp_vl.v0_info.virtual_numbers, cudaMemcpyHostToDevice);
            if (temp_vl.v1_info.virtual_numbers>0)
                cudaMemcpy(temp_vl.v1_info.d_virtual_type_1, temp_vl.v1_info.h_virtual_type_1, sizeof(VIRTUAL_TYPE_1) * temp_vl.v1_info.virtual_numbers, cudaMemcpyHostToDevice);
            if (temp_vl.v2_info.virtual_numbers>0)
                cudaMemcpy(temp_vl.v2_info.d_virtual_type_2, temp_vl.v2_info.h_virtual_type_2, sizeof(VIRTUAL_TYPE_2)* temp_vl.v2_info.virtual_numbers, cudaMemcpyHostToDevice);
            if (temp_vl.v3_info.virtual_numbers>0)
                cudaMemcpy(temp_vl.v3_info.d_virtual_type_3, temp_vl.v3_info.h_virtual_type_3, sizeof(VIRTUAL_TYPE_3)* temp_vl.v3_info.virtual_numbers, cudaMemcpyHostToDevice);
        }
        controller[0].printf("    End reading information for every virtual atom\n");

        is_initialized = 1;
        if (is_initialized && !is_controller_printf_initialized)
        {
            is_controller_printf_initialized = 1;
            controller[0].printf("    structure last modify date is %d\n", last_modify_date);
        }

        controller[0].printf("END INITIALIZING VIRTUAL ATOM\n\n");
    }
    else
    {
        controller->printf("VIRTUAL ATOM IS NOT INITIALIZED\n\n");
    }
}

void VIRTUAL_INFORMATION::Coordinate_Refresh(const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, VECTOR *crd)
{
    if (is_initialized)
    {
        //每层之间需要串行计算，层内并行计算
        for (int layer = 0; layer < max_level; layer++)
        {
            VIRTUAL_LAYER_INFORMATION temp_vl = virtual_layer_info[layer];
            if (temp_vl.v0_info.virtual_numbers>0)
                v0_Coordinate_Refresh << <(unsigned int)ceilf((float)temp_vl.v0_info.virtual_numbers / threads_per_block), threads_per_block >> > (temp_vl.v0_info.virtual_numbers, temp_vl.v0_info.d_virtual_type_0, uint_crd, scaler, crd);
            if (temp_vl.v1_info.virtual_numbers>0)
                v1_Coordinate_Refresh << <(unsigned int)ceilf((float)temp_vl.v1_info.virtual_numbers / threads_per_block), threads_per_block >> > (temp_vl.v1_info.virtual_numbers, temp_vl.v1_info.d_virtual_type_1, uint_crd, scaler, crd);
            if (temp_vl.v2_info.virtual_numbers>0)
                v2_Coordinate_Refresh << <(unsigned int)ceilf((float)temp_vl.v2_info.virtual_numbers / threads_per_block), threads_per_block >> > (temp_vl.v2_info.virtual_numbers, temp_vl.v2_info.d_virtual_type_2, uint_crd, scaler, crd);
            if (temp_vl.v3_info.virtual_numbers>0)
                v3_Coordinate_Refresh << <(unsigned int)ceilf((float)temp_vl.v3_info.virtual_numbers / threads_per_block), threads_per_block >> > (temp_vl.v3_info.virtual_numbers, temp_vl.v3_info.d_virtual_type_3, uint_crd, scaler, crd);
        }
    }
}

void VIRTUAL_INFORMATION::Force_Redistribute(const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, VECTOR *frc)
{
    if (is_initialized)
    {
        //每层之间需要串行逆向计算，层内并行计算
        for (int layer = max_level - 1; layer >= 0; layer--)
        {
            VIRTUAL_LAYER_INFORMATION temp_vl = virtual_layer_info[layer];
            if (temp_vl.v0_info.virtual_numbers>0)
                v0_Force_Redistribute << <(unsigned int)ceilf((float)temp_vl.v0_info.virtual_numbers / threads_per_block), threads_per_block >> > (temp_vl.v0_info.virtual_numbers, temp_vl.v0_info.d_virtual_type_0, uint_crd, scaler, frc);
            if (temp_vl.v1_info.virtual_numbers>0)
                v1_Force_Redistribute << <(unsigned int)ceilf((float)temp_vl.v1_info.virtual_numbers / threads_per_block), threads_per_block >> > (temp_vl.v1_info.virtual_numbers, temp_vl.v1_info.d_virtual_type_1, uint_crd, scaler, frc);
            if (temp_vl.v2_info.virtual_numbers>0)
                v2_Force_Redistribute << <(unsigned int)ceilf((float)temp_vl.v2_info.virtual_numbers / threads_per_block), threads_per_block >> > (temp_vl.v2_info.virtual_numbers, temp_vl.v2_info.d_virtual_type_2, uint_crd, scaler, frc);
            if (temp_vl.v3_info.virtual_numbers>0)
                v3_Force_Redistribute << <(unsigned int)ceilf((float)temp_vl.v3_info.virtual_numbers / threads_per_block), threads_per_block >> > (temp_vl.v3_info.virtual_numbers, temp_vl.v3_info.d_virtual_type_3, uint_crd, scaler, frc);
        }
    }
}

void VIRTUAL_INFORMATION::Clear()
{
    if (is_initialized)
    {
        is_initialized = 0;
        for (int layer = 0; layer < max_level; layer++)
        {
            VIRTUAL_LAYER_INFORMATION *temp_vl = &virtual_layer_info[layer];
            if (temp_vl[0].v0_info.virtual_numbers > 0)
            {
                free(temp_vl[0].v0_info.h_virtual_type_0);
                cudaFree(temp_vl[0].v0_info.d_virtual_type_0);
            }
            if (temp_vl[0].v1_info.virtual_numbers > 0)
            {
                free(temp_vl[0].v1_info.h_virtual_type_1);
                cudaFree(temp_vl[0].v1_info.d_virtual_type_1);
            }
            if (temp_vl[0].v2_info.virtual_numbers > 0)
            {
                free(temp_vl[0].v2_info.h_virtual_type_2);
                cudaFree(temp_vl[0].v2_info.d_virtual_type_2);
            }
            if (temp_vl[0].v3_info.virtual_numbers > 0)
            {
                free(temp_vl[0].v3_info.h_virtual_type_3);
                cudaFree(temp_vl[0].v3_info.d_virtual_type_3);
            }
        }
        free(virtual_level);
        virtual_layer_info.clear();
    }
}
