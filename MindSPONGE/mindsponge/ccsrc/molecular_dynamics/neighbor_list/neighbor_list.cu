#include "neighbor_list.cuh"

static void Initial_Neighbor_Grid(
    GRID_POINTER **gpointer, GRID_BUCKET **bucket, int **atom_numbers_in_grid_bucket,
    float half_cutoff_with_skin, GRID_INFORMATION *grid_info,
    const int in_bucket_atom_numbers_max, VECTOR box_length)
{


    float half_cutoff = half_cutoff_with_skin;

    grid_info[0].Nx = floorf(box_length.x / half_cutoff);
    grid_info[0].Ny = floorf(box_length.y / half_cutoff);
    grid_info[0].Nz = floorf(box_length.z / half_cutoff);
    grid_info[0].grid_N = { grid_info[0].Nx, grid_info[0].Ny, grid_info[0].Nz };

    grid_info[0].grid_length.x = (float)box_length.x / grid_info[0].Nx;
    grid_info[0].grid_length_inverse.x = 1. / grid_info[0].grid_length.x;
    grid_info[0].grid_length.y = (float)box_length.y / grid_info[0].Ny;
    grid_info[0].grid_length_inverse.y = 1. / grid_info[0].grid_length.y;
    grid_info[0].grid_length.z = (float)box_length.z / grid_info[0].Nz;
    grid_info[0].grid_length_inverse.z = 1. / grid_info[0].grid_length.z;

    grid_info[0].Nxy = grid_info[0].Nx*grid_info[0].Ny;
    grid_info[0].grid_numbers = grid_info[0].Nz*grid_info[0].Nxy;

    Cuda_Malloc_Safely((void **)&atom_numbers_in_grid_bucket[0], sizeof(int)*(grid_info[0].grid_numbers+1));
    Reset_List << <ceilf(((float)grid_info[0].grid_numbers + 1 ) / 32), 32 >> >(grid_info[0].grid_numbers+1, atom_numbers_in_grid_bucket[0], 0);

    Malloc_Safely((void**)&grid_info[0].h_bucket,sizeof(GRID_BUCKET)*(grid_info[0].grid_numbers+1)); 
    for (int i = 0; i < grid_info[0].grid_numbers + 1; i = i + 1)
    {
        Cuda_Malloc_Safely((void**)&grid_info[0].h_bucket[i].atom_serial, sizeof(int)* in_bucket_atom_numbers_max);
        Reset_List << <ceilf((float)in_bucket_atom_numbers_max/32), 32 >> >(in_bucket_atom_numbers_max, grid_info[0].h_bucket[i].atom_serial, -1);
    }
    Cuda_Malloc_Safely((void**)&bucket[0], sizeof(GRID_BUCKET)*(grid_info[0].grid_numbers+1));
    cudaMemcpy(bucket[0], grid_info[0].h_bucket, sizeof(GRID_BUCKET)*(grid_info[0].grid_numbers+1), cudaMemcpyHostToDevice);
    //free(h_bucket);

    GRID_POINTER lin_pointer;
    lin_pointer.grid_serial = (int*)malloc(sizeof(int)* 125);
    int Nx;
    int Ny;
    int Nz;
    int xx;
    int yy;
    int zz;
    int count;
    int small_out;
    Malloc_Safely((void**)&grid_info[0].h_pointer, sizeof(GRID_POINTER)*grid_info[0].grid_numbers);
    for (int i = 0; i < grid_info[0].grid_numbers; i = i + 1)
    {
        Nz = i / grid_info[0].Nxy;
        Ny = (i - grid_info[0].Nxy*Nz) / grid_info[0].Nx;
        Nx = i - grid_info[0].Nxy*Nz - grid_info[0].Nx*Ny;
        count = 0;
        for (int l = -2; l <= 2; l = l + 1)
        {
            for (int m = -2; m <= 2; m = m + 1)
            {
                for (int n = -2; n <= 2; n = n + 1)
                {
                    small_out = 0;
                    xx = Nx + l;
                    //处理小盒子越边界
                    //盒子大小大于5、未超过边界、等于4且只超出一个边界的，不处理
                    if (grid_info->Nx >= 5 || (xx >= 0 && xx < grid_info->Nx) || (grid_info->Nx == 4 && ((Nx == 0 && l == -1) || (Nx == 3 && l == 1))))
                    {

                    }
                    else
                    {
                        small_out = 1;
                    }
                    if (!small_out)
                    {
                        if (xx < 0)
                        {
                            xx = xx + grid_info[0].Nx;
                        }
                        else if (xx >= grid_info[0].Nx)
                        {
                            xx = xx - grid_info[0].Nx;
                        }
                    }
                    yy = Ny + m;
                    //处理小盒子越边界
                    if (grid_info->Ny >= 5 || (yy >= 0 && yy < grid_info->Ny) || (grid_info->Ny == 4 && ((Ny == 0 && m == -1) || (Ny == 3 && m == 1))))
                    {

                    }
                    else
                    {
                        small_out = 1;
                    }
                    if (!small_out)
                    {
                        if (yy < 0)
                        {
                            yy = yy + grid_info[0].Ny;
                        }
                        else if (yy >= grid_info[0].Ny)
                        {
                            yy = yy - grid_info[0].Ny;
                        }
                    }
                    
                    zz = Nz + n;
                    //处理小盒子越边界
                    if (grid_info->Nz >= 5 || (zz >= 0 && zz < grid_info->Nz) || (grid_info->Nz == 4 && ((Nz == 0 && n == -1) || (Nz == 3 && n == 1))))
                    {

                    }
                    else
                    {
                        small_out = 1;
                    }
                    if (!small_out)
                    {
                        if (zz < 0)
                        {
                            zz = zz + grid_info[0].Nz;
                        }
                        else if (zz >= grid_info[0].Nz)
                        {
                            zz = zz - grid_info[0].Nz;
                        }
                    }

                    if (!small_out)
                    {
                        lin_pointer.grid_serial[count] = zz*grid_info[0].Nxy + yy*grid_info[0].Nx + xx;
                        
                    }
                    else
                    {
                        lin_pointer.grid_serial[count] = grid_info->grid_numbers;
                    }
                    count = count + 1;
                }
            }
        }//for l m n
        thrust::sort(&lin_pointer.grid_serial[0], lin_pointer.grid_serial + 125);
        Cuda_Malloc_Safely((void**)&grid_info[0].h_pointer[i].grid_serial, sizeof(int)* 125);//5*5*5
        cudaMemcpy(grid_info[0].h_pointer[i].grid_serial, lin_pointer.grid_serial, sizeof(int)* 125, cudaMemcpyHostToDevice);
    }
    Cuda_Malloc_Safely((void**)&gpointer[0], sizeof(GRID_POINTER)*grid_info[0].grid_numbers);
    
    cudaMemcpy(gpointer[0], grid_info[0].h_pointer, sizeof(GRID_POINTER)*grid_info[0].grid_numbers, cudaMemcpyHostToDevice);
}

static __global__ void Clear_Grid_Bucket(const int grid_numbers, int *atom_numbers_in_grid_bucket, GRID_BUCKET *bucket)
{
    int grid_serial = blockDim.x*blockIdx.x + threadIdx.x;
    if (grid_serial < grid_numbers)
    {
        GRID_BUCKET bucket_i = bucket[grid_serial];
        for (int i = 0; i < atom_numbers_in_grid_bucket[grid_serial]; i = i + 1)
        {
            bucket_i.atom_serial[i] = -1;
        }
        atom_numbers_in_grid_bucket[grid_serial] = 0;
    }
}

static __global__ void Find_Atom_In_Grid_Serial(const int atom_numbers, const VECTOR grid_length_inverse, const VECTOR *crd, const INT_VECTOR grid_N, const int gridxy, int *atom_in_grid_serial)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        int Nx = (float)crd[atom_i].x*grid_length_inverse.x;//crd.x must < boxlength.x
        int Ny = (float)crd[atom_i].y*grid_length_inverse.y;
        int Nz = (float)crd[atom_i].z*grid_length_inverse.z;
        /*Nx = min(Nx, 8);
        Ny = min(Ny, 8);
        Nz = min(Nz, 8);*/
        Nx = Nx&((Nx - grid_N.int_x) >> 31);
        Ny = Ny&((Ny - grid_N.int_y) >> 31);
        Nz = Nz&((Nz - grid_N.int_z) >> 31);
        atom_in_grid_serial[atom_i] = Nz*gridxy + Ny*grid_N.int_x + Nx;
        //20210417 debug
        //if (atom_in_grid_serial[atom_i] >= 729 || atom_in_grid_serial[atom_i] < 0)
        //{
        //    atom_in_grid_serial[atom_i] = 0;
        //    printf("fuck %d %d %f %f %f\n",atom_i, atom_in_grid_serial[atom_i], crd[atom_i].x, crd[atom_i].y, crd[atom_i].z);
        //}
    }
}
static __global__ void Put_Atom_In_Grid_Bucket(const int atom_numbers, const int *atom_in_grid_serial, GRID_BUCKET *bucket, int *atom_numbers_in_grid_bucket)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        int grid_serial = atom_in_grid_serial[atom_i];
        GRID_BUCKET bucket_i = bucket[grid_serial];
        int a = atom_numbers_in_grid_bucket[grid_serial];
        atomicCAS(&bucket_i.atom_serial[a], -1, atom_i);
        if (bucket_i.atom_serial[a] != atom_i)
        {
            while (true)
            {
                a = a + 1;
                //20210417 debug
                //if (a >= 64)
                //{
                //    break;
                //}
                atomicCAS(&bucket_i.atom_serial[a], -1, atom_i);
                if (bucket_i.atom_serial[a] == atom_i)
                {
                    atomicAdd(&atom_numbers_in_grid_bucket[grid_serial], 1);
                    break;
                }
            }
        }
        else
        {
            atomicAdd(&atom_numbers_in_grid_bucket[grid_serial], 1);
        }
    }
}

static __global__ void Find_atom_neighbors_gridly(
    const int atom_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR uint_dr_to_dr_cof,
    const int *atom_in_grid_serial, const GRID_POINTER *gpointer, const GRID_BUCKET *bucket, const int *atom_numbers_in_grid_bucket,
    ATOM_GROUP *nl, const float cutoff_skin_square, const int max_atom_numbers_in_gird)
{
    if (threadIdx.y < 125)
    {
        int grid_i = blockIdx.x;
        int grid_j = gpointer[grid_i].grid_serial[threadIdx.y];
        int atom_i, atom_j;
        VECTOR dr;
        float dr2;
        extern __shared__ char shared_memory[];

        UNSIGNED_INT_VECTOR *uints = (UNSIGNED_INT_VECTOR *)shared_memory;
        int *sm_bucket_i = (int*)(shared_memory + sizeof(UNSIGNED_INT_VECTOR)* max_atom_numbers_in_gird);
        ATOM_GROUP *sm_nl = (ATOM_GROUP *)(shared_memory + (sizeof(UNSIGNED_INT_VECTOR)+sizeof(int))* max_atom_numbers_in_gird);

        int *bucket_i = bucket[grid_i].atom_serial;
        int *bucket_j = bucket[grid_j].atom_serial;
        int atom_numbers_in_grid_i = atom_numbers_in_grid_bucket[grid_i];
        int atom_numbers_in_grid_j = atom_numbers_in_grid_bucket[grid_j];
        if (threadIdx.x == 0)
        {
            for (int i = threadIdx.y; i < atom_numbers_in_grid_i; i += blockDim.y)
            {
                atom_i = bucket_i[i];
                uints[i] = uint_crd[atom_i];
                sm_bucket_i[i] = atom_i;        
                sm_nl[i] = nl[atom_i];
                nl[atom_i].atom_numbers = 0;
            }
        }
        __syncthreads();
    
        UNSIGNED_INT_VECTOR uint_crd_j;
        ATOM_GROUP nl_i;
        for (int j = threadIdx.x; j < atom_numbers_in_grid_j; j += blockDim.x)
        {
            atom_j = bucket_j[j];
            uint_crd_j = uint_crd[atom_j];

            for (int i = 0; i < atom_numbers_in_grid_i; i++)
            {
                atom_i = sm_bucket_i[i];
                nl_i = sm_nl[i];

                if (atom_j > atom_i)
                {            
                    dr = Get_Periodic_Displacement(uint_crd_j, uints[i], uint_dr_to_dr_cof);
                    dr2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
                    if (dr2 < cutoff_skin_square)
                    {
                        nl_i.atom_numbers = atomicAdd(&nl[atom_i].atom_numbers, 1);
                        nl_i.atom_serial[nl_i.atom_numbers] = atom_j;
                    }
                }
            }
        }
    }
}
/*
static __global__ void Find_atom_neighbors(
    const int atom_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR uint_dr_to_dr_cof,
    const int *atom_in_grid_serial, const GRID_POINTER *gpointer, const GRID_BUCKET *bucket, const int *atom_numbers_in_grid_bucket,
    ATOM_GROUP *nl, const float cutoff_skin_square)
{
    int atom_i = blockDim.y*blockIdx.y + threadIdx.y;
    if (atom_i < atom_numbers)
    {
        int grid_serial = atom_in_grid_serial[atom_i];
        int grid_serial2;
        
        int atom_j;
        int int_x;
        int int_y;
        int int_z;
        UNSIGNED_INT_VECTOR uint_crd_i = uint_crd[atom_i], uint_crd_j;
        ATOM_GROUP nl_i = nl[atom_i];
        int *atom_numbers_address = &nl[atom_i].atom_numbers;
        *atom_numbers_address = 0;
        __syncthreads();
        GRID_POINTER gpointer_i = gpointer[grid_serial];
        VECTOR dr;
        float dr2;
        for (int grid_cycle = threadIdx.x; grid_cycle < 125; grid_cycle = grid_cycle + blockDim.x)
        {
            grid_serial2 = gpointer_i.grid_serial[grid_cycle];
            GRID_BUCKET bucket_i = bucket[grid_serial2];
            for (int j = 0; j < atom_numbers_in_grid_bucket[grid_serial2]; j = j + 1)
            {
                atom_j = bucket_i.atom_serial[j];
                uint_crd_j = uint_crd[atom_j];
                if (atom_j > atom_i)
                {
                    int_x = uint_crd_j.uint_x - uint_crd_i.uint_x;
                    int_y = uint_crd_j.uint_y - uint_crd_i.uint_y;
                    int_z = uint_crd_j.uint_z - uint_crd_i.uint_z;
                    dr.x = uint_dr_to_dr_cof.x*int_x;
                    dr.y = uint_dr_to_dr_cof.y*int_y;
                    dr.z = uint_dr_to_dr_cof.z*int_z;
                    dr2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
                    if (dr2 < cutoff_skin_square)
                    {
                        nl_i.atom_numbers = atomicAdd(atom_numbers_address, 1);
                        nl_i.atom_serial[nl_i.atom_numbers] = atom_j;
                    }
                }
            }
        }//125 grid cycle
    }
}
*/
static __global__ void Is_need_refresh_neighbor_list_cuda(const int atom_numbers,const VECTOR *crd, const VECTOR *old_crd,
    const VECTOR box_length, const float half_skin_square,int *need_refresh_flag)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < atom_numbers)
    {
        VECTOR r1 = crd[i];
        VECTOR r2 = old_crd[i];
        r1 = Get_Periodic_Displacement(r1, r2, box_length);
        float r1_2 = r1.x*r1.x + r1.y*r1.y + r1.z*r1.z;
        if (r1_2>half_skin_square)
        {
            atomicExch(&need_refresh_flag[0], 1);
        }
    }
}

static __global__ void Delete_Excluded_Atoms_Serial_In_Neighbor_List
(const int atom_numbers, ATOM_GROUP *nl, const int *excluded_list_start,const int *excluded_list,const int *excluded_atom_numbers)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        int excluded_number = excluded_atom_numbers[atom_i];
        if (excluded_number > 0 )
        {
            int list_start = excluded_list_start[atom_i];
            int atom_min = excluded_list[list_start];
            int list_end = list_start + excluded_number;
            int atom_max = excluded_list[list_end - 1];
            ATOM_GROUP nl_i = nl[atom_i];
            int atomnumbers_in_nl_lin = nl_i.atom_numbers;
            int atom_j;
            int excluded_atom_numbers_lin = list_end-list_start;
            int excluded_atom_numbers_count = 0;
            for (int i = 0; i < atomnumbers_in_nl_lin; i = i + 1)
            {
                atom_j = nl_i.atom_serial[i];
                if (atom_j<atom_min || atom_j>atom_max)
                {
                    continue;
                }
                else
                {
                    for (int j = list_start; j < list_end; j = j + 1)
                    {
                        if (atom_j == excluded_list[j])
                        {
                            atomnumbers_in_nl_lin = atomnumbers_in_nl_lin - 1;
                            nl_i.atom_serial[i] = nl_i.atom_serial[atomnumbers_in_nl_lin];
                            excluded_atom_numbers_count = excluded_atom_numbers_count + 1;
                            i=i-1;
                        }
                    }
                    if (excluded_atom_numbers_count < excluded_atom_numbers_lin)
                    {
                        ;
                    }
                    else
                    {
                        break;
                    }//break
                }//in the range of excluded min to max
            }//cycle for neighbors
            nl[atom_i].atom_numbers = atomnumbers_in_nl_lin;
        }//if need excluded
    }
}

static __global__ void Refresh_Neighbor_List
(int *refresh_sign, const int thread,
const int atom_numbers, VECTOR *crd, VECTOR *old_crd, UNSIGNED_INT_VECTOR *uint_crd,
const VECTOR quarter_crd_to_uint_crd_cof, const VECTOR uint_dr_to_dr_cof,
int *atom_in_grid_serial,
const float skin, const VECTOR box_length,
const GRID_INFORMATION grid_info, const GRID_POINTER *gpointer,
GRID_BUCKET *bucket, int *atom_numbers_in_grid_bucket,
ATOM_GROUP *d_nl, int *excluded_list_start, int * excluded_list, int * excluded_numbers, float cutoff_skin_square, const int max_atom_in_grid_numbers)
{
    if (refresh_sign[0] == 1)
    {

        Clear_Grid_Bucket << <ceilf((float)grid_info.grid_numbers / thread), thread >> >
            (grid_info.grid_numbers, atom_numbers_in_grid_bucket, bucket);

        Crd_Periodic_Map << <ceilf((float)atom_numbers / thread), thread >> >(atom_numbers, crd, box_length);

        Find_Atom_In_Grid_Serial << <ceilf((float)atom_numbers / thread), thread >> >
            (atom_numbers, grid_info.grid_length_inverse, crd, grid_info.grid_N, grid_info.Nxy, atom_in_grid_serial);

        Copy_List << <ceilf((float)3.*atom_numbers / thread), thread >> >
            (3 * atom_numbers, (float*)crd, (float*)old_crd);

        Put_Atom_In_Grid_Bucket << <ceilf((float)atom_numbers / thread), thread >> >
            (atom_numbers, atom_in_grid_serial, bucket, atom_numbers_in_grid_bucket);

        Crd_To_Uint_Crd << <ceilf((float)atom_numbers / thread), thread >> >
            (atom_numbers, quarter_crd_to_uint_crd_cof, crd, uint_crd);
        
        Find_atom_neighbors_gridly << < {(unsigned int)ceilf((float)grid_info.grid_numbers)}, { 8, 128 }, (sizeof(int)+sizeof(UNSIGNED_INT_VECTOR)+sizeof(ATOM_GROUP))*max_atom_in_grid_numbers >> >
            (atom_numbers, uint_crd, uint_dr_to_dr_cof,
            atom_in_grid_serial, gpointer, bucket, atom_numbers_in_grid_bucket,
            d_nl, cutoff_skin_square, max_atom_in_grid_numbers);
        
        /*Find_atom_neighbors << < {1, (unsigned int)ceilf((float)atom_numbers / 125)}, { 8, 125 } >> >
            (atom_numbers, uint_crd, uint_dr_to_dr_cof,
            atom_in_grid_serial, gpointer, bucket, atom_numbers_in_grid_bucket,
            d_nl, cutoff_skin_square);
*/
        

        Delete_Excluded_Atoms_Serial_In_Neighbor_List << <ceilf((float)atom_numbers / thread), thread >> >
            (atom_numbers, d_nl, excluded_list_start, excluded_list, excluded_numbers);
        refresh_sign[0] = 0;
    }
}


static void Refresh_Neighbor_List_No_Check
(const int atom_numbers, VECTOR *crd, VECTOR *old_crd, UNSIGNED_INT_VECTOR *uint_crd,
const VECTOR quarter_crd_to_uint_crd_cof, const VECTOR uint_dr_to_dr_cof,
int *atom_in_grid_serial,
const float skin, const VECTOR box_length,
const GRID_INFORMATION grid_info, const GRID_POINTER *gpointer,
GRID_BUCKET *bucket, int *atom_numbers_in_grid_bucket,
ATOM_GROUP *d_nl, int *excluded_list_start, int * excluded_list, int * excluded_numbers, float cutoff_skin_square, const int max_atom_in_grid_numbers)
{
    Clear_Grid_Bucket << <ceilf((float)grid_info.grid_numbers / 32), 32 >> >
        (grid_info.grid_numbers, grid_info.atom_numbers_in_grid_bucket, grid_info.bucket);

    Crd_Periodic_Map << <ceilf((float)atom_numbers / 32), 32 >> >(atom_numbers, crd, box_length);

    Find_Atom_In_Grid_Serial << <ceilf((float)atom_numbers / 32), 32 >> >
        (atom_numbers, grid_info.grid_length_inverse, crd, grid_info.grid_N, grid_info.Nxy, grid_info.atom_in_grid_serial);

    cudaMemcpy(old_crd, crd, sizeof(VECTOR)*atom_numbers, cudaMemcpyDeviceToDevice);

    Put_Atom_In_Grid_Bucket << <ceilf((float)atom_numbers / 32), 32 >> >
        (atom_numbers, grid_info.atom_in_grid_serial, grid_info.bucket, grid_info.atom_numbers_in_grid_bucket);

    Crd_To_Uint_Crd << <ceilf((float)atom_numbers / 32), 32 >> >
        (atom_numbers, quarter_crd_to_uint_crd_cof, crd, uint_crd);

    
    Find_atom_neighbors_gridly << < {(unsigned int)ceilf((float)grid_info.grid_numbers)}, { 8, 128 }, (sizeof(int)+sizeof(UNSIGNED_INT_VECTOR)+sizeof(ATOM_GROUP))*max_atom_in_grid_numbers >> >
        (atom_numbers, uint_crd, uint_dr_to_dr_cof,
        atom_in_grid_serial, gpointer, bucket, atom_numbers_in_grid_bucket,
        d_nl, cutoff_skin_square, max_atom_in_grid_numbers);

    /*
    Find_atom_neighbors << < {1, (unsigned int)ceilf((float)atom_numbers / 125)}, { 8, 125 } >> >
        (atom_numbers, uint_crd, uint_dr_to_dr_cof,
        atom_in_grid_serial, gpointer, bucket, atom_numbers_in_grid_bucket,
        d_nl, cutoff_skin_square);*/
/*    ATOM_GROUP *temp;
    int *atom_temp;
    Malloc_Safely((void**)&temp, sizeof(ATOM_GROUP));
    Malloc_Safely((void**)&atom_temp, sizeof(int) * 800);
    for (int i = 98274; i < atom_numbers; i++)
    {
        cudaMemcpy(temp, d_nl + i, sizeof(ATOM_GROUP), cudaMemcpyDeviceToHost);
        cudaMemcpy(atom_temp, temp->atom_serial, sizeof(int) * 800, cudaMemcpyDeviceToHost);
        for (int j = 0; j < temp->atom_numbers; j++)
        {
            printf("%d %d %d\n", i, j, atom_temp[j]);
        }
    }
    getchar();*/    
    Delete_Excluded_Atoms_Serial_In_Neighbor_List << <ceilf((float)atom_numbers / 32), 32 >> >
        (atom_numbers, d_nl, excluded_list_start, excluded_list, excluded_numbers);
}



void NEIGHBOR_LIST::Neighbor_List_Update(VECTOR *crd, int *d_excluded_list_start, int *d_excluded_list, int *d_excluded_numbers,
    int forced_update, int forced_check)
{
    if (is_initialized)
    {    
        if (forced_update) //如果强制要求更新就强制更新
        {
            Refresh_Neighbor_List_No_Check
                (atom_numbers, crd, old_crd, uint_crd,
                quarter_crd_to_uint_crd_cof, uint_dr_to_dr_cof,
                grid_info.atom_in_grid_serial,
                skin, box_length,
                grid_info, grid_info.gpointer,
                grid_info.bucket, grid_info.atom_numbers_in_grid_bucket,
                d_nl, d_excluded_list_start, d_excluded_list, d_excluded_numbers, cutoff_with_skin_square, max_atom_in_grid_numbers);
        }
        else if (refresh_interval > 0 && !forced_check) //如果是恒步长更新且不强制要求检查是否更新
        {
            if (refresh_count % refresh_interval == 0)
            {
                Refresh_Neighbor_List_No_Check
                    (atom_numbers, crd, old_crd, uint_crd,
                    quarter_crd_to_uint_crd_cof, uint_dr_to_dr_cof,
                    grid_info.atom_in_grid_serial,
                    skin, box_length,
                    grid_info, grid_info.gpointer,
                    grid_info.bucket, grid_info.atom_numbers_in_grid_bucket,
                    d_nl, d_excluded_list_start, d_excluded_list, d_excluded_numbers, cutoff_with_skin_square, max_atom_in_grid_numbers);
            }
            refresh_count += 1;
        }
        else //其余情况
        {
            Is_need_refresh_neighbor_list_cuda << <ceilf((float)atom_numbers / 128), 128 >> >
                (atom_numbers, crd, old_crd, box_length, skin_permit*skin_permit*half_skin_square, is_need_refresh_neighbor_list);
            Refresh_Neighbor_List << <1, 1 >> >
                (is_need_refresh_neighbor_list, 32,
                atom_numbers, crd, old_crd, uint_crd,
                quarter_crd_to_uint_crd_cof, uint_dr_to_dr_cof,
                grid_info.atom_in_grid_serial,
                skin, box_length,
                grid_info, grid_info.gpointer,
                grid_info.bucket, grid_info.atom_numbers_in_grid_bucket,
                d_nl, d_excluded_list_start, d_excluded_list, d_excluded_numbers, cutoff_with_skin_square, max_atom_in_grid_numbers);
        }
    }
}

void NEIGHBOR_LIST::Initial_Malloc()
{
    Cuda_Malloc_Safely((void **)&old_crd, sizeof(VECTOR)*atom_numbers);
    Cuda_Malloc_Safely((void **)&uint_crd, sizeof(UNSIGNED_INT_VECTOR)*atom_numbers);
    Malloc_Safely((void**)&h_nl, sizeof(ATOM_GROUP)*atom_numbers);
    Cuda_Malloc_Safely((void**)&d_nl, sizeof(ATOM_GROUP)*atom_numbers);
    for (int i = 0; i < atom_numbers; i = i + 1)
    {
        h_nl[i].atom_numbers = 0;
        Cuda_Malloc_Safely((void**)&h_nl[i].atom_serial, sizeof(int)* max_neighbor_numbers);
    }
    cudaMemcpy(d_nl, h_nl, sizeof(ATOM_GROUP)*atom_numbers, cudaMemcpyHostToDevice);
    for (int i = 0; i < atom_numbers; i = i + 1)
    {
        Malloc_Safely((void**)&h_nl[i].atom_serial, sizeof(int)* max_neighbor_numbers);
    }
    Cuda_Malloc_Safely((void**)&is_need_refresh_neighbor_list, sizeof(int));
    Reset_List << <1, 1 >> >(1, is_need_refresh_neighbor_list, 0);
    Cuda_Malloc_Safely((void**)&grid_info.atom_in_grid_serial, sizeof(int)*atom_numbers);
}


void NEIGHBOR_LIST::Initial(CONTROLLER *controller, int md_atom_numbers, VECTOR box_length, float cut, float skin, const char * module_name)
{
    
    if (module_name == NULL)
    {
        strcpy(this->module_name, "neighbor_list");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    /*===========================
    从mdin中读取控制信息
    ============================*/
    controller[0].printf("START INITIALIZING NEIGHBOR LIST:\n");
    atom_numbers = md_atom_numbers;
    if (controller[0].Command_Exist(this->module_name, "refresh_interval"))
    {
        refresh_interval = atoi(controller[0].Command(this->module_name, "refresh_interval"));
    }
    if (controller[0].Command_Exist(this->module_name, "max_atom_in_grid_numbers"))
    {
        max_atom_in_grid_numbers = atoi(controller[0].Command(this->module_name, "max_atom_in_grid_numbers"));
    }
    if (controller[0].Command_Exist(this->module_name, "max_neighbor_numbers"))
    {
        max_neighbor_numbers = atoi(controller[0].Command(this->module_name, "max_neighbor_numbers"));
    }
    if (controller[0].Command_Exist(this->module_name, "skin_permit"))
    {
        skin_permit = 2.*atof(controller[0].Command(this->module_name, "skin_permit"));//以外界的0.5等于不变（即程序内的1.）
    }
    this->skin = skin;
    this->cutoff = cut;
    cutoff_square = cutoff*cutoff;
    cutoff_with_skin = cutoff + skin;
    half_cutoff_with_skin = 0.5*cutoff_with_skin;
    cutoff_with_skin_square = cutoff_with_skin*cutoff_with_skin;
    half_skin_square = 0.25*skin*skin;
        

    this->box_length = box_length;
    this->quarter_crd_to_uint_crd_cof = 0.25f * CONSTANT_UINT_MAX_FLOAT / box_length;
    this->uint_dr_to_dr_cof = 1.0f / CONSTANT_UINT_MAX_FLOAT * box_length;


    /*===========================
    //初始化格子信息
    ============================*/
    Initial_Malloc();
    Initial_Neighbor_Grid(
        &grid_info.gpointer, &grid_info.bucket, &grid_info.atom_numbers_in_grid_bucket,
        half_cutoff_with_skin, &grid_info,
        max_atom_in_grid_numbers, box_length);
    is_initialized = 1;
    controller->printf("    grid dimension is %d %d %d\n", grid_info.Nx, grid_info.Ny, grid_info.Nz);
    if (is_initialized && !is_controller_printf_initialized)
    {
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }
    controller[0].printf("END INITIALIZING NEIGHBOR LIST\n\n");
}


void NEIGHBOR_LIST::Update_Volume(VECTOR box_length)
{
    this->box_length = box_length;
    this->quarter_crd_to_uint_crd_cof = 0.25f * CONSTANT_UINT_MAX_FLOAT/ box_length;
    this->uint_dr_to_dr_cof = 1.0f / CONSTANT_UINT_MAX_FLOAT * box_length;

    grid_info.grid_length.x = (float)box_length.x / grid_info.Nx;
    grid_info.grid_length.y = (float)box_length.y / grid_info.Ny;
    grid_info.grid_length.z = (float)box_length.z / grid_info.Nz;

    grid_info.grid_length_inverse = 1.0f / grid_info.grid_length;
}


void NEIGHBOR_LIST::Clear()
{
    if (is_initialized == 1)
    {
        is_initialized = 0;
        cudaFree(old_crd);
        old_crd = NULL;
        cudaFree(uint_crd);
        uint_crd = NULL;
        cudaFree(is_need_refresh_neighbor_list);
        is_need_refresh_neighbor_list = NULL;

        for (int i = 0; i < atom_numbers; i = i + 1)
        {
            free(h_nl[i].atom_serial);
        }
        cudaMemcpy(h_nl, d_nl, sizeof(ATOM_GROUP)*atom_numbers, cudaMemcpyDeviceToHost);
        for (int i = 0; i < atom_numbers; i = i + 1)
        {
            cudaFree(h_nl[i].atom_serial);
        }
        free(h_nl);
        h_nl = NULL;
        cudaFree(d_nl);
        d_nl = NULL;

        cudaFree(grid_info.atom_in_grid_serial);
        grid_info.atom_in_grid_serial = NULL;

        cudaFree(grid_info.atom_numbers_in_grid_bucket);
        grid_info.atom_numbers_in_grid_bucket = NULL;

        cudaMemcpy(grid_info.h_bucket, grid_info.bucket, sizeof(GRID_BUCKET)*grid_info.grid_numbers, cudaMemcpyDeviceToHost);
        for (int i = 0; i < grid_info.grid_numbers; i = i + 1)
        {
            cudaFree(grid_info.h_bucket[i].atom_serial);
        }
        cudaFree(grid_info.bucket);
        grid_info.bucket = NULL;
        free(grid_info.h_bucket);
        grid_info.h_bucket = NULL;

        cudaMemcpy(grid_info.h_pointer, grid_info.gpointer, sizeof(GRID_BUCKET)*grid_info.grid_numbers, cudaMemcpyDeviceToHost);
        for (int i = 0; i < grid_info.grid_numbers; i = i + 1)
        {
            cudaFree(grid_info.h_pointer[i].grid_serial);
        }
        cudaFree(grid_info.gpointer);
        grid_info.gpointer = NULL;
        free(grid_info.h_pointer);
        grid_info.h_pointer = NULL;
    }
}
