#include "constrain.cuh"


void CONSTRAIN::Initial_Constrain(CONTROLLER *controller, const int atom_numbers, const float dt, const VECTOR box_length, const float exp_gamma, const int is_Minimization, float *atom_mass, int *system_freedom)
{
    //从传入的参数复制基本信息
    this->atom_numbers = atom_numbers;
    this->dt = dt;
    this->dt_inverse = 1.0 / dt;
    this->quarter_crd_to_uint_crd_cof = 0.25 * CONSTANT_UINT_MAX_FLOAT / box_length;
    this->uint_dr_to_dr_cof = 1.0f / CONSTANT_UINT_MAX_FLOAT * box_length;
    this->volume = box_length.x * box_length.y * box_length.z;


    //确定使用的朗之万热浴，并给定exp_gamma
    this->v_factor = exp_gamma;
    this->x_factor = 0.5*(1. + exp_gamma);

    if (is_Minimization)
    {
        this->v_factor = 0.0f;
    }

    int extra_numbers = 0;
    FILE *fp = NULL;
    //读文件第一个数确认constrain数量，为分配内存做准备
    if (controller[0].Command_Exist(this->module_name, "in_file"))
    {
        Open_File_Safely(&fp, controller[0].Command(this->module_name, "in_file"), "r");
        int scanf_ret = fscanf(fp, "%d", &extra_numbers);
    }

    constrain_pair_numbers = bond_constrain_pair_numbers + angle_constrain_pair_numbers + extra_numbers;
    system_freedom[0] -= constrain_pair_numbers;
    controller[0].printf("    constrain pair number is %d\n", constrain_pair_numbers);

    Malloc_Safely((void**)&h_constrain_pair, sizeof(CONSTRAIN_PAIR)*constrain_pair_numbers);
    Cuda_Malloc_Safely((void**)&constrain_pair, sizeof(CONSTRAIN_PAIR)*constrain_pair_numbers);
    for (int i = 0; i < bond_constrain_pair_numbers; i = i + 1)
    {
        h_constrain_pair[i] = h_bond_pair[i];
        h_constrain_pair[i].constrain_k = h_constrain_pair[i].constrain_k / this->x_factor;
    }
    for (int i = 0; i < angle_constrain_pair_numbers; i = i + 1)
    {
        h_constrain_pair[i + bond_constrain_pair_numbers] = h_angle_pair[i];
        h_constrain_pair[i + bond_constrain_pair_numbers].constrain_k = h_constrain_pair[i + bond_constrain_pair_numbers].constrain_k / this->x_factor;
    }
    //读文件存入
    if (fp != NULL)
    {
        int atom_i, atom_j;
        int count = bond_constrain_pair_numbers + angle_constrain_pair_numbers;
        for (int i = 0; i < extra_numbers; i = i + 1)
        {
            int scanf_ret = fscanf(fp, "%d %d %f", &atom_i, &atom_j, &h_constrain_pair[count].constant_r);
            h_constrain_pair[count].atom_i_serial = atom_i;
            h_constrain_pair[count].atom_j_serial = atom_j;
            h_constrain_pair[count].constrain_k = atom_mass[atom_i] * atom_mass[atom_j] / (atom_mass[atom_i] + atom_mass[atom_j]) / this->x_factor;
            count += 1;
        }
        fclose(fp);
        fp = NULL;
    }

    //上传GPU
    cudaMemcpy(constrain_pair, h_constrain_pair, sizeof(CONSTRAIN_PAIR)*constrain_pair_numbers, cudaMemcpyHostToDevice);


    //清空初始化时使用的临时变量
    if (h_bond_pair != NULL)
    {
        free(h_bond_pair);
        h_bond_pair = NULL;
    }
    if (h_angle_pair != NULL)
    {
        free(h_angle_pair);
        h_angle_pair = NULL;
    }

    if (is_initialized && !is_controller_printf_initialized)
    {
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }
    controller[0].printf("END INITIALIZING CONSTRAIN\n\n");
    is_initialized = 1;
}


void CONSTRAIN::Add_HBond_To_Constrain_Pair
(CONTROLLER *controller, const int bond_numbers, const int *atom_a, const int *atom_b, const float *bond_r,
const float *atom_mass, const char *module_name)
{
    controller[0].printf("START INITIALIZING CONSTRAIN:\n");
    if (module_name == NULL)
    {
        strcpy(this->module_name, "constrain");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    constrain_mass = 3.3f;
    if (controller[0].Command_Exist(this->module_name,"in_file"))
        constrain_mass = 0.0f;
    if (controller[0].Command_Exist(this->module_name, "mass"))
        constrain_mass = atof(controller[0].Command(this->module_name, "mass"));
    //预先分配一个足够大的CONSTRAIN_PAIR用于临时存储
    Malloc_Safely((void**)&h_bond_pair, sizeof(CONSTRAIN_PAIR)*bond_numbers);
    int s = 0;
    float mass_a, mass_b;
    for (int i = 0; i < bond_numbers; i = i + 1)
    {
        mass_a = atom_mass[atom_a[i]];
        mass_b = atom_mass[atom_b[i]];
        if ((mass_a <constrain_mass && mass_a > 0) || (mass_b <constrain_mass && mass_b > 0))//含有H原子的bond
        {
            h_bond_pair[s].atom_i_serial = atom_a[i];
            h_bond_pair[s].atom_j_serial = atom_b[i];
            h_bond_pair[s].constant_r = bond_r[i];
            h_bond_pair[s].constrain_k = atom_mass[atom_a[i]] * atom_mass[atom_b[i]] / (atom_mass[atom_a[i]] + atom_mass[atom_b[i]]);
            s = s + 1;
        }
    }
    bond_constrain_pair_numbers = s;

    //假设使用者不会在调用Add_HBond_To_Constrain_Pair与Add_HAngle_To_Constrain_Pair中途释放atom_a等指针指向的空间
    bond_info.bond_numbers = bond_numbers;
    bond_info.atom_a = atom_a;
    bond_info.atom_b = atom_b;
    bond_info.bond_r = bond_r;
}
void CONSTRAIN::Add_HAngle_To_Constrain_Pair
(CONTROLLER *controller, const int angle_numbers, const int *atom_a, const int *atom_b, const int *atom_c,
const float *angle_theta, const float *atom_mass)
{
    int temp = 0;
    if (controller->Command_Exist(this->module_name, "angle") && atoi(controller->Command(this->module_name, "angle")))
    {
        temp = angle_numbers;
    }
    
    //默认认为已经运行了Add_HBond_To_Constrain_Pair

    //预先分配一个足够大的CONSTRAIN_PAIR用于临时存储
    Malloc_Safely((void**)&h_angle_pair, sizeof(CONSTRAIN_PAIR)*angle_numbers*2);
    int s = 0;
    float mass_a, mass_c;
    for (int i = 0; i < temp; i = i + 1)
    {
        mass_a = atom_mass[atom_a[i]];
        mass_c = atom_mass[atom_c[i]];
        if ((mass_a <constrain_mass && mass_a > 0) || (mass_c <constrain_mass && mass_c > 0))//含有H原子的angle,假设氢原子不会在角中心。
        {
            h_angle_pair[s].atom_i_serial = atom_a[i];//固定angle两端的两个点
            h_angle_pair[s].atom_j_serial = atom_c[i];

            float rab=0., rbc=0.;
            for (int j = 0; j < bond_info.bond_numbers; j = j + 1)
            {
                //找到a，b原子的平衡距离
                if ((bond_info.atom_a[j] == atom_a[i] && bond_info.atom_b[j] == atom_b[i])
                    || (bond_info.atom_a[j] == atom_b[i] && bond_info.atom_b[j] == atom_a[i]))
                {
                    rab = bond_info.bond_r[j];
                }

                //找到b，c原子的平衡距离
                if ((bond_info.atom_a[j] == atom_c[i] && bond_info.atom_b[j] == atom_b[i])
                    || (bond_info.atom_a[j] == atom_b[i] && bond_info.atom_b[j] == atom_c[i]))
                {
                    rbc = bond_info.bond_r[j];
                }
            }
            if (rab == 0. || rbc == 0.)
            {
                controller[0].printf("    Error: Wrong BOND and ANGLE combination!\n");
                getchar();
                continue;
            }

            //运用余弦定理得到平衡的ac长度用于constrain
            h_angle_pair[s].constant_r = sqrtf(rab*rab + rbc*rbc - 2.*rab*rbc*cosf(angle_theta[i]));
            h_angle_pair[s].constrain_k = atom_mass[atom_a[i]] * atom_mass[atom_c[i]] / (atom_mass[atom_a[i]] + atom_mass[atom_c[i]]);

            s = s + 1;

        }
    }
    angle_constrain_pair_numbers = s;
}

void CONSTRAIN::Update_Volume(VECTOR box_length)
{
    if (is_initialized)
    {
        quarter_crd_to_uint_crd_cof = 0.25 * CONSTANT_UINT_MAX_FLOAT / box_length;
        uint_dr_to_dr_cof = 1.0f / CONSTANT_UINT_MAX_FLOAT * box_length;
        volume = box_length.x * box_length.y * box_length.z;
    }
}

void CONSTRAIN::Clear()
{
    if (is_initialized)
    {
        is_initialized = 0;

        cudaFree(constrain_pair);
        constrain_pair = NULL;

        free(h_constrain_pair);
        h_constrain_pair = NULL;
    }
}
