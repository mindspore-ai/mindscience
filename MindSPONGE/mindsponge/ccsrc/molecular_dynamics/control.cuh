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


#ifndef CONTROL_CUH
#define CONTROL_CUH
#include "map"
#include "string.h"
#include "common.cuh"
#include "vector"
#include "stdarg.h"

typedef std::map<std::string, std::string> StringMap;
typedef std::map<std::string, int> CheckMap;
typedef std::vector<std::string> StringVector;

//用于记录时间
struct TIME_RECORDER
{
private:
    clock_t start_timestamp;
    clock_t end_timestamp;
public:
    double time = 0;
    void Start();
    void Stop();
    void Clear();
};

struct CONTROLLER
{
    //自身信息
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20210827;

    void Initial(int argc, char **argv);
    void Clear();

    //输入控制部分
    StringMap original_commands;  //由文件读入的原始命令
    StringMap commands;   //由文件读入的命令（去除空格）
    CheckMap command_check;   //检查输入是否都使用了
    CheckMap choice_check;   //检查选项是否都使用了
    void Get_Command(char *line, char *prefix); //内部解析argument时专用，设置命令，不外部调用
    void Set_Command(const char *Flag, const char *Value, int Check = 1, const char *prefix = NULL); //内部解析argument时专用，设置命令，不外部调用
    void Arguments_Parse(int argc, char **argv);  //对终端输入进行分析
    void Commands_From_In_File(int argc, char **argv); //对mdin输入进行分析
    void Default_Set();  //对最基本的功能进行默认设置
    void Init_Cuda();   //对cuda初始化
    //本部分的上面的内容最好不要外部调用

    void Input_Check(); //检查所有输入是否都被使用了（防止错误的输入未被检查到）

    bool Command_Exist(const char *key);   //判断文件读入的命令中是否有key
    bool Command_Exist(const char *prefix, const char *key);   //判断文件读入的命令中是否有key
    //判断是否存在key且值为value。未设置时返回no_set_return_value，可控制比对是否对大小写敏感
    bool Command_Choice(const char *key, const char *value, bool case_sensitive = 0);
    //判断是否存在prefix_key且值为value。未设置时返回no_set_return_value，可控制比对是否对大小写敏感
    bool Command_Choice(const char *prefix, const char *key, const char *value, bool case_sensitive = 0);
    const char * Command(const char *key);   //获得文件读入的命令key对应的value
    const char * Command(const char *prefix, const char *key);   //获得文件读入的命令key对应的value
    const char * Original_Command(const char *key);   //获得文件读入的命令key对应的value
    const char * Original_Command(const char *prefix, const char *key);   //获得文件读入的命令key对应的value

    //计时部分
    TIME_RECORDER core_time; //计时器
    float simulation_speed; //模拟运行速度（纳秒/天）

    //输出控制部分
    FILE *mdinfo = NULL;  //屏幕信息打印文件
    FILE *mdout = NULL;
    StringMap outputs_content;  //记录每步输出数值
    StringMap outputs_format; //记录每步输出的格式
    StringVector outputs_key; //记录每部输出的表头
    //本部分的上面的内容最好不要外部调用

    void printf(const char *fmt, ...);  //重载printf，使得printf能够同时打印到mdinfo和屏幕
    void Step_Print_Initial(const char *head, const char *format); //其他模块初始化时调用，获得对应的表头和格式
    void Step_Print(const char *head, const float *pointer); //其他模块打印时调用，获得对应的表头和数值，使之能以不同的格式打印在屏幕和mdout
    void Step_Print(const char *head, const float pointer);  //其他模块打印时调用，获得对应的表头和数值，使之能以不同的格式打印在屏幕和mdout
    void Step_Print(const char *head, const double pointer);  //其他模块打印时调用，获得对应的表头和数值，使之能以不同的格式打印在屏幕和mdout
    void Step_Print(const char *head, const int pointer);    //其他模块打印时调用，获得对应的表头和数值，使之能以不同的格式打印在屏幕和mdout
    void Print_First_Line_To_Mdout(FILE *mdout = NULL);             //模拟开始前的操作，将表头打印到mdout，并在屏幕打印一个分割线
    void Print_To_Screen_And_Mdout(FILE *mdout = NULL);             //模拟开始每步的调用，使得其他部分的结果打印到屏幕和mdout
};
//判断两个字符串是否相等（无视大小写）
bool is_str_equal(const char* a_str, const char *b_str, int case_sensitive = 0);

#endif //CONTROL_CUH(control.cuh)
