#ifndef NEIGHBOR_LIST_CUH
#define NEIGHBOR_LIST_CUH
#include "../common.cuh"
#include "../control.cuh"

//GRID相关结构体只是为了方便近邻表给空间打格点，外部一般不需要调用
//用于记录某个grid周围的125个grid（包含自身）的指针
struct GRID_POINTER
{
    int *grid_serial = NULL;//5*5*5
};
//用于记录某个grid中含有的原子序号列表
struct GRID_BUCKET
{
    int *atom_serial = NULL;//32 may be enough
};

//用于记录构造近邻表所需要使用的grid信息
struct GRID_INFORMATION
{
    int grid_numbers;//总grid数目,在Initial_Neighbor_Grid中进行初始化
    int Nx;
    int Ny;
    int Nxy;
    int Nz;
    INT_VECTOR grid_N;//就等于Nx Ny Nz组成的一个整型向量
    VECTOR grid_length;
    VECTOR grid_length_inverse;

    int *atom_in_grid_serial = NULL;//每个原子所在grid的编号
    int *atom_numbers_in_grid_bucket = NULL;//每个bucket所容纳的原子个数

    GRID_POINTER *gpointer = NULL;//指向每个grid的125个邻居grid（包含自身）的指针
    GRID_BUCKET *bucket = NULL;//每个grid的容器指针

    GRID_POINTER *h_pointer = NULL;//指向每个grid的125个邻居grid（包含自身）的指针
    GRID_BUCKET *h_bucket = NULL;//每个grid的容器指针

    int block_for_device;
    int thread_for_device;
};

//用于记录近邻表相关信息
struct NEIGHBOR_LIST
{
    //自身信息
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20210826;

    int atom_numbers = 0;//整个模拟体系的原子数目,跟随MD信息
    
    int refresh_interval = 0;
    int refresh_count = 0;

    float skin = 2.0;//近邻表外圈厚度,跟随MD信息
    float cutoff = 10.0;//LJ的截断半径,跟随MD信息
    float cutoff_square;//在Initial_neighbor_list中初始化
    float cutoff_with_skin;
    float half_cutoff_with_skin;
    float cutoff_with_skin_square;
    float half_skin_square;//单次近邻表更新时，原子能运动的最大距离平方
    float skin_permit=1.;//用于不严格要求的智能更新近邻表修正，该值越大，更新信号越不容易被触发。
    VECTOR box_length;
    VECTOR quarter_crd_to_uint_crd_cof;
    VECTOR uint_dr_to_dr_cof;

    int refresh_flag;//指示更新近邻表，在CPU上使用
    int *is_need_refresh_neighbor_list = NULL; //用于指示是否需要更新近邻表，在GPU上使用
    
    VECTOR *old_crd = NULL;//用于判断是否更新近邻表
    UNSIGNED_INT_VECTOR *uint_crd = NULL; //用于寻找近邻
    ATOM_GROUP *h_nl = NULL;//在host上的近邻表，基本只在初始化时使用
    ATOM_GROUP *d_nl = NULL;//在devicce上的近邻表
    int max_neighbor_numbers = 800;    //每个原子的最大近邻数


    GRID_INFORMATION grid_info;
    int max_atom_in_grid_numbers = 64; //每个格点可以放的最大原子数

    //初始化
    void Initial(CONTROLLER *controller, int md_atom_numbers, VECTOR box_length, float cut, float skin, const char *module_name = NULL);
    //分配内存
    void Initial_Malloc();
    //清除内存
    void Clear();

    //更新近邻表，需要：浮点数坐标，整数坐标，1/4浮到整转换系数，整到浮转换系数，盒子的大小，排除表相关信息，是否强制更新，是否强制检查
    void Neighbor_List_Update(VECTOR *crd, int *d_excluded_list_start, int *d_excluded_list, int *d_excluded_numbers,
        int forced_update = 0, int forced_check = 0);

    void Update_Volume(VECTOR box_length);

    //去除魔鬼数字用
    enum NEIGHBOR_LIST_UPDATE_PARAMETER
    {
        CONDITIONAL_UPDATE = 0,
        FORCED_UPDATE = 1
    };
    enum NEIGHBOR_LIST_CHECK_PARAMETER
    {
        CONDITIONAL_CHECK = 0,
        FORCED_CHECK = 1
    };
};

#endif //NEIGHBOR_LIST_CUH(neighbor_list.cuh)
