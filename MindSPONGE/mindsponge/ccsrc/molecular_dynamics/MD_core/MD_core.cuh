/*
 * Copyright 2021 Gao's lab, Peking University, CCME. All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MD_CORE_CUH
#define MD_CORE_CUH
#include <deque>
#include "../common.cuh"
#include "../control.cuh"

//普通分子模拟所涉及的大部分信息
struct MD_INFORMATION {
  int is_initialized = 0;
  int last_modify_date = 20211105;

  // sponge输入初始化
  void Initial(CONTROLLER *controller);

  char md_name[CHAR_LENGTH_MAX];
  int mode = 0; // md的模式(-1: 最小化, 0: NVE, 1: NVT, 2: NPT)
  enum MD_MODE { RERUN = -2, MINIMIZATION = -1, NVE = 0, NVT = 1, NPT = 2 };
  void Read_Mode(CONTROLLER *controller); //读取mode

  float dt =
      0.001 *
      CONSTANT_TIME_CONVERTION; //模拟步长，单位为(0.0488878 ps)==(1/20.455 ps)
  void Read_dt(CONTROLLER *controller); //读取dt

  int atom_numbers = 0; //模拟的总原子数目
  //每个原子的基本物理测量量，on host
  VECTOR *velocity = NULL;
  VECTOR *coordinate = NULL;
  VECTOR *force = NULL;
  VECTOR *acceleration = NULL;
  float *h_mass = NULL;
  float *h_mass_inverse = NULL;
  float *h_charge = NULL;

  //每个原子的基本物理测量量，on device
  VECTOR *vel = NULL;
  VECTOR *crd = NULL;
  UNSIGNED_INT_VECTOR *uint_crd = NULL; //用于快速周期性映射
  VECTOR *frc = NULL;
  VECTOR *acc = NULL;
  float *d_mass = NULL;
  float *d_mass_inverse = NULL;
  float *d_charge = NULL;
  //坐标读取处理
  void Read_Coordinate_And_Velocity(CONTROLLER *controller);
  //读rst7文件获得坐标、速度（可选）、系统时间、盒子
  void Read_Rst7(const char *file_name, int irest, CONTROLLER controller);
  //读坐标文件获得坐标、速度（可选）、系统时间、盒子
  void Read_Coordinate_In_File(const char *file_name, CONTROLLER controller);
  //读取质量
  void Read_Mass(CONTROLLER *controller);
  //读取电荷
  void Read_Charge(CONTROLLER *controller);

  //每个原子的能量和维里相关
  int need_pressure = 0;
  int need_potential = 0;
  float *h_atom_energy = NULL;
  float *h_atom_virial = NULL;
  float *d_atom_energy = NULL;
  float *d_atom_virial = NULL;
  float *d_atom_ek = NULL;
  //为结构体中的数组变量分配存储空间
  void Atom_Information_Initial();
  //计算力前将原子能量和维里和力归零（如果需要计算时）
  void MD_Reset_Atom_Energy_And_Virial_And_Force();
  //通过原子势能、原子维里、原子动能计算压强和总势能到GPU上（如果需要）
  void Calculate_Pressure_And_Potential_If_Needed(int is_download = 1);

  struct system_information {
    MD_INFORMATION *md_info = NULL;
    int freedom = 0;         //体系自由度
    int steps = 0;           //当前模拟的步数
    int step_limit = 1000;   //需要模拟的步数
    double start_time = 0;   //系统初始时间 ps
    double dt_in_ps = 0.001; //用于记录时间所用的dt
    double current_time = 0; //系统现在时间 ps
    double Get_Current_Time();

    float total_mass = 0; //总质量 道尔顿
    VECTOR box_length;    //模拟体系的边界大小 angstrom
    float volume = 0;     //体积 angstrom^3
    float Get_Volume();

    float density = 0; //密度 g/cm^3
    float Get_Density();

    float h_virial;         //系统总标量维里 kcal/mol
    float *d_virial = NULL; //系统总标量维里 kcal/mol

    float *d_pressure = NULL; //体系压强 系统单位
    float h_pressure;         //体系压强 系统单位
    float target_pressure;    //外界压浴压强 系统单位

    float Get_Pressure(int is_download = 1);

    float h_potential;         //体系势能
    float *d_potential = NULL; //体系势能
    float Get_Potential(int is_download = 1);

    float h_sum_of_atom_ek;         //体系原子动能
    float *d_sum_of_atom_ek = NULL; //体系原子动能
    float Get_Total_Atom_Ek(int is_download = 1);

    float h_temperature;          //体系温度 K
    float *d_temperature = NULL;  //体系温度 K
    float target_temperature;     //外界热浴温度 K
    float Get_Atom_Temperature(); //自由度还有问题

    void Initial(CONTROLLER *controller, MD_INFORMATION *md_info);
  } sys; // 系统整体信息
  struct non_bond_information {
    MD_INFORMATION *md_info = NULL;
    float cutoff = 10.0;
    float skin = 2.0;
    int excluded_atom_numbers;  //排除表总长
    int *d_excluded_list_start; //记录每个原子的剔除表起点
    int *d_excluded_list;       //剔除表
    int *d_excluded_numbers;    //记录每个原子需要剔除的原子个数
    int *h_excluded_list_start; //记录每个原子的剔除表起点
    int *h_excluded_list;       //剔除表
    int *h_excluded_numbers;    //记录每个原子需要剔除的原子个数
    void Initial(CONTROLLER *controller, MD_INFORMATION *md_info);
  } nb; // 非键信息
  struct periodic_box_condition_information {
    VECTOR crd_to_uint_crd_cof;         //实坐标到整数坐标
    VECTOR quarter_crd_to_uint_crd_cof; //实坐标到0.25倍整数坐标
    VECTOR uint_dr_to_dr_cof;           //整数坐标到实坐标
    void Initial(CONTROLLER *controller, VECTOR box_length);
  } pbc;
  struct trajectory_output {
    MD_INFORMATION *md_info = NULL;
    int current_crd_synchronized_step = 0;
    int is_molecule_map_output = 0;
    int amber_irest = -1;
    int write_trajectory_interval = 1000; //打印轨迹内容的所隔步数
    int write_mdout_interval = 1000;      //打印能量信息的所隔步数
    int write_restart_file_interval = 1000; // restart文件重新创建的所隔步数
    FILE *crd_traj = NULL;
    FILE *box_traj = NULL;
    char restart_name[CHAR_LENGTH_MAX];
    void Initial(CONTROLLER *controller, MD_INFORMATION *md_info);
    void Export_Restart_File(const char *rst7_name = NULL);
    void Append_Crd_Traj_File(FILE *fp = NULL);
    void Append_Box_Traj_File(FILE *fp = NULL);
    // 20210827用于输出速度和力
    int is_frc_traj = 0, is_vel_traj = 0;
    FILE *frc_traj = NULL;
    FILE *vel_traj = NULL;
    void Append_Frc_Traj_File(FILE *fp = NULL);
    void Append_Vel_Traj_File(FILE *fp = NULL);
  } output; //轨迹输出信息
  struct NVE_iteration {
    MD_INFORMATION *md_info =
        NULL; //指向自己主结构体的指针，以方便调用主结构体的信息
    float max_velocity = -1;
    void Leap_Frog();
    void Velocity_Verlet_1();
    void Velocity_Verlet_2();
    void Initial(CONTROLLER *controller, MD_INFORMATION *md_info);
  } nve; // nve迭代相关函数

  struct MINIMIZATION_iteration {
    MD_INFORMATION *md_info =
        NULL; //指向自己主结构体的指针，以方便调用主结构体的信息
    float max_move = 0.02;
    int dynamic_dt = 0;
    float last_potential = 0;
    float momentum_keep = 0;
    float dt_increasing_rate = 1.01;
    float dt_decreasing_rate = 0.01;
    void Gradient_Descent();
    void Initial(CONTROLLER *controller, MD_INFORMATION *md_info);
  } min;

  struct RERUN_information {
    MD_INFORMATION *md_info =
        NULL; //指向自己主结构体的指针，以方便调用主结构体的信息
    FILE *traj_file = NULL;
    FILE *box_file = NULL;
    VECTOR box_length_change_factor;
    void Initial(CONTROLLER *controller, MD_INFORMATION *md_info);
    void Iteration();
  } rerun;

  struct residue_information {
    int is_initialized = 0;
    MD_INFORMATION *md_info =
        NULL; //指向自己主结构体的指针，以方便调用主结构体的信息
    int residue_numbers = 0; //模拟的总残基数目

    float *h_mass = NULL;         //残基质量
    float *h_mass_inverse = NULL; //残基质量的倒数
    int *h_res_start = NULL;      //残基起始编号
    int *h_res_end = NULL;    //残基终止编号（实际为终止编号+1）
    float *h_momentum = NULL; //残基动量
    VECTOR *h_center_of_mass = NULL; //残基质心
    float *h_sigma_of_res_ek = NULL; //残基平动能求和

    float *res_ek_energy = NULL; //残基平动能（求温度时已乘系数）

    float *sigma_of_res_ek = NULL; //残基平动能求和
    int *d_res_start = NULL;       //残基起始编号
    int *d_res_end = NULL; //残基终止编号（实际为终止编号+1）
    float *d_mass = NULL;  //残基质量
    float *d_mass_inverse = NULL;    //残基质量的倒数
    float *d_momentum = NULL;        //残基动量
    VECTOR *d_center_of_mass = NULL; //残基质心
    void Residue_Crd_Map(
        VECTOR *no_wrap_crd,
        float scaler =
            1.0f); //将坐标质心映射到盒子中，且如果scaler>0则乘上scaler

    void Initial(CONTROLLER *controller, MD_INFORMATION *md_info);
    void Read_AMBER_Parm7(const char *file_name, CONTROLLER controller);
    float Get_Total_Residue_Ek(int is_download = 1);
    float h_temperature; //残基平动温度 K
    float Get_Residue_Temperature();
  } res; //残基信息

  struct molecule_information {
    int is_initialized = 0;
    MD_INFORMATION *md_info =
        NULL; //指向自己主结构体的指针，以方便调用主结构体的信息
    int molecule_numbers = 0; //模拟的总分子数目

    float *h_mass = NULL;         //分子质量
    float *h_mass_inverse = NULL; //分子质量的倒数
    int *h_atom_start = NULL;     //分子起始的原子编号
    int *h_atom_end = NULL; //分子终止的原子编号（实际为终止编号+1）
    int *h_residue_start = NULL; //分子起始的残基编号
    int *h_residue_end = NULL; //分子终止的残基编号（实际为终止编号+1）
    VECTOR *h_center_of_mass = NULL; //分子质心

    int *d_atom_start = NULL; //分子起始的原子编号
    int *d_atom_end = NULL; //分子终止的原子编号（实际为终止编号+1）
    int *d_residue_start = NULL; //分子起始的残基编号
    int *d_residue_end = NULL; //分子终止的残基编号（实际为终止编号+1）
    float *d_mass = NULL;            //分子质量
    float *d_mass_inverse = NULL;    //分子质量的倒数
    VECTOR *d_center_of_mass = NULL; //分子质心

    void Molecule_Crd_Map(
        VECTOR *no_wrap_crd,
        float scaler =
            1.0f); //将坐标质心映射到盒子中，且如果scaler>0则乘上scaler
    void Molecule_Crd_Map(
        VECTOR *no_wrap_crd,
        VECTOR scaler); //将坐标质心映射到盒子中，且如果scaler>0则乘上scaler

    void Initial(CONTROLLER *controller, MD_INFORMATION *md_info);
  } mol; //分子信息

  //体积变化一个因子
  void Update_Volume(double factor);

  //体积变化一个因子
  void Update_Box_Length(VECTOR factor);

  //用来将原子的真实坐标转换为unsigned
  //int坐标,注意factor需要乘以0.5（保证越界坐标自然映回box）
  void MD_Information_Crd_To_Uint_Crd();

  //将frc拷贝到cpu上
  void MD_Information_Frc_Device_To_Host();

  //将force拷贝到gpu上
  void MD_Information_Frc_Host_To_Device();

  //将crd拷贝到cpu上
  void Crd_Vel_Device_To_Host(int Do_Translation = 1, int forced = 0);

  //释放空间
  void Clear();
};

#endif // MD_CORE_CUH(MD_core.cuh)
