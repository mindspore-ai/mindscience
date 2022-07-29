#include "control.cuh"

#define SPONGE_VERSION "v1.2.6.0 2022-08-01"

#define MDIN_DEFAULT_FILENAME "mdin.txt"
#define MDOUT_DEFAULT_FILENAME "mdout.txt"
#define MDINFO_DEFAULT_FILENAME "mdinfo.txt"

#define MDIN_COMMAND "mdin"
#define MDOUT_COMMAND "mdout"
#define MDINFO_COMMAND "mdinfo"


bool is_str_equal(const char* a_str, const char *b_str, int case_sensitive)
{
    int i = 0;
    char a;
    char b;
    while (true)
    {
        if (a_str[i] == 0 && b_str[i] == 0)
        {
            return 1;
        }
        else if (a_str[i] == 0 || b_str[i] == 0)
        {
            return 0;
        }
        else
        {
            a = a_str[i];
            b = b_str[i];
            if (!case_sensitive)
            {
                if (a >= 65 && a <= 90)
                {
                    a = a - 65 + 97;
                }
                if (b >= 65 && b <= 90)
                {
                    b = b - 65 + 97;
                }
            }
            if (a != b)
            {
                return 0;
            }        
        }
        i = i + 1;
    }
}
bool CONTROLLER::Command_Exist(const char *key)
{
    const char *temp = strstr(key, "in_file");
    command_check[key] = 0;
    if (temp != NULL && strcmp(temp, "in_file") == 0)
    {
        if (commands.count(key))
        {
            return true;
        }
        else if (Command_Exist("default_in_file_prefix"))
        {
            
            char buffer[CHAR_LENGTH_MAX], buffer2[CHAR_LENGTH_MAX];
            strcpy(buffer, key);
            
            buffer[strlen(key) - strlen(temp) - 1] = 0;
            sprintf(buffer2, "%s_%s.txt", Command("default_in_file_prefix"), buffer);
            FILE *ftemp = fopen(buffer2, "r");
            if (ftemp != NULL)
            {
                commands[key] = buffer2;
                fclose(ftemp);
                return true;
            }
            return false;

        }
        else
        {
            return false;
        }
    }
    else
    {
        return (bool)commands.count(key);
    }
}

bool CONTROLLER::Command_Exist(const char *prefix, const char *key)
{
    char temp[CHAR_LENGTH_MAX];
    strcpy(temp, prefix);
    strcat(temp, "_");
    strcat(temp, key);
    return Command_Exist(temp);
}

bool CONTROLLER::Command_Choice(const char *key, const char *value, bool case_sensitive)
{
    if (commands.count(key))
    {
        if (is_str_equal(commands[key].c_str(), value, case_sensitive))
        {
            command_check[key] = 0;
            choice_check[key] = 1;
            return true;
        }
        else
        {
            command_check[key] = 0;
            if (choice_check[key] != 1)
                choice_check[key] = 2;
            return false;
        }
    }
    else
    {
        choice_check[key] = 3;
                return false;
    }
}

bool CONTROLLER::Command_Choice(const char *prefix, const char *key, const char *value, bool case_sensitive)
{
    char temp[CHAR_LENGTH_MAX];
    strcpy(temp, prefix);
    strcat(temp, "_");
    strcat(temp, key);
    return Command_Choice(temp, value, case_sensitive);
}

const char * CONTROLLER::Command(const char *key)
{
    command_check[key] = 0;
    return commands[key].c_str();
}

const char * CONTROLLER::Command(const char *prefix, const char *key)
{
    char temp[CHAR_LENGTH_MAX];
    strcpy(temp, prefix);
    strcat(temp, "_");
    strcat(temp, key);
    command_check[temp] = 0;
    return commands[temp].c_str();
}

const char * CONTROLLER::Original_Command(const char *key)
{
    command_check[key] = 0;
    return original_commands[key].c_str();
}

const char * CONTROLLER::Original_Command(const char *prefix, const char *key)
{
    char temp[CHAR_LENGTH_MAX];
    strcpy(temp, prefix);
    strcat(temp, "_");
    strcat(temp, key);
    command_check[temp] = 0;
    return original_commands[temp].c_str();
}

static int judge_if_flag(const char *str)
{
    if (strlen(str) <= 1)
        return 0;
    if (str[0] != '-')
        return 0;
    if (str[1] >= '0' && str[1] <='9')
        return 0;
    return 1;
}
void CONTROLLER::Arguments_Parse(int argc, char **argv)
{
    char temp1[CHAR_LENGTH_MAX];
    char temp2[CHAR_LENGTH_MAX];
    char temp3[CHAR_LENGTH_MAX];
    int j = 1;
    for (int i = 1; i < argc; i++)
    {
        temp1[0] = 0;
        strcat(temp1, argv[i]);
        if (judge_if_flag(temp1))
        {
            temp2[0] = 0;
            j = 1;
            while (i + j < argc)
            {
                strcpy(temp3, argv[i + j]);
                if (!judge_if_flag(temp3))
                {
                    strcat(temp2, " ");
                    strcat(temp2, temp3);
                    j++;
                }
                else
                    break;
            }
            Set_Command(temp1+1, temp2);
        }
    }
}

void CONTROLLER::Get_Command(char *line, char *prefix)
{
    
    if ((prefix[0] == '#' && prefix[1] == '#') || prefix[0] == ' ' || prefix[0] == '\t')
    {
        return;
    }
    char Flag[CHAR_LENGTH_MAX];
    char Value[CHAR_LENGTH_MAX];
    char *flag = strtok(line, "=");
    char *command = strtok(NULL, "=");
    
    if (flag == NULL || command == NULL)
    {
        return;
    }
    
    sscanf(flag, "%s", Flag);
    strcpy(Value, command);
    //fprintf(stdout, "%s|\n%s|\n%s|\n\n", Flag, Value, prefix); //debug用
    Set_Command(Flag, Value, 1, prefix);

}

static int read_one_line(FILE *In_File, char *line, char *ender)
{
    int line_count = 0;
    int ender_count = 0;
    char c;
    while ((c = getc(In_File)) != EOF)
    {
        if (line_count == 0 && (c == '\t' || c == ' '))
        {
            continue;
        }
        else if (c != '\n' && c != ',' && c != '{' && c != '}')
        {
            line[line_count] = c;
            line_count += 1;
        }
        else
        {
            ender[ender_count] = c;
            ender_count += 1;
            break;
        }
    }
    while ((c = getc(In_File)) != EOF)
    {
        if (c == ' ' || c == '\t')
        {
            continue;
        }
        else if (c != '\n' && c != ',' && c != '{' && c != '}')
        {
            fseek(In_File, -1, SEEK_CUR);
            break;
        }
        else
        {
            ender[ender_count] = c;
            ender_count += 1;
        }
    }
    line[line_count] = 0;
    ender[ender_count] = 0;
    if (line_count == 0 && ender_count == 0)
    {
        return EOF;
    }
    return 1;
}

void CONTROLLER::Commands_From_In_File(int argc, char **argv)
{


    FILE *In_File = NULL;
    if (!Command_Exist(MDIN_COMMAND))
    {
        In_File = fopen(MDIN_DEFAULT_FILENAME, "r");
        if (In_File == NULL)
        {
            commands["md_name"] = "Default SPONGE MD Task Name";
        }
    }
    else
        Open_File_Safely(&In_File, Command(MDIN_COMMAND), "r");
    if (In_File != NULL)
    {
        char line[CHAR_LENGTH_MAX];
        char prefix[CHAR_LENGTH_MAX] = { 0 };
        char ender[CHAR_LENGTH_MAX];
        char *get_ret = fgets(line, CHAR_LENGTH_MAX, In_File);
        line[strlen(line) - 1] = 0;
        commands["md_name"] = line;
        while (true)
        {
            if (read_one_line(In_File, line, ender) == EOF)
            {
                break;
            }
            //printf("%s\n%s\n", line, ender);
            if (line[0] == '#')
            {
                if (line[1] == '#')
                {
                    if (strchr(ender, '{') != NULL)
                    {
                        int scanf_ret = sscanf(line, "%s", prefix);
                    }
                    if (strchr(ender, '}') != NULL )
                    {
                        prefix[0] = 0;
                    }
                }
                if (strchr(ender, '\n') == NULL)                
                {
                    int scanf_ret = fscanf(In_File, "%*[^\n]%*[\n]");
                    fseek(In_File, -1, SEEK_CUR);
                }
            }
            else if (strchr(ender, '{') != NULL)
            {
                int scanf_ret = sscanf(line, "%s", prefix);
            }
            else
            {
                Get_Command(line, prefix);
                line[0] = 0;
            }
            if (strchr(ender, '}') != NULL)
            {
                prefix[0] = 0;
            }
        }
    }

    if (Command_Exist(MDINFO_COMMAND))
    {
        Open_File_Safely(&mdinfo, Command(MDINFO_COMMAND), "w");
    }
    else
    {
        Open_File_Safely(&mdinfo, MDINFO_DEFAULT_FILENAME, "w");
    }
    if (Command_Exist(MDOUT_COMMAND))
    {
        Open_File_Safely(&mdout, Command(MDOUT_COMMAND), "w");
    }
    else
    {
        Open_File_Safely(&mdout, MDOUT_DEFAULT_FILENAME, "w");
    }
    printf("SPONGE Version:\n    %s\n\n", SPONGE_VERSION);
    printf("Citation:\n    %s\n", "Huang, Y. - P., Xia, Y., Yang, L., Wei, J., Yang, Y.I.and Gao, Y.Q. (2022), SPONGE: A GPU - Accelerated Molecular Dynamics Package with Enhanced Sampling and AI - Driven Algorithms.Chin.J.Chem., 40 : 160 - 168. https ://doi.org/10.1002/cjoc.202100456\n\n");
    printf("MD TASK NAME:\n    %s\n\n", commands["md_name"].c_str());
    int scanf_ret = fprintf(mdinfo, "Terminal Commands:\n    ");
    for (int i = 0; i < argc; i++)
    {
        scanf_ret = fprintf(mdinfo, "%s ", argv[i]);
    }
    scanf_ret = fprintf(mdinfo, "\n\n");
    if (In_File != NULL)
    {
        scanf_ret = fprintf(mdinfo, "Mdin File:\n");
        fseek(In_File, 0, SEEK_SET);
        char temp[CHAR_LENGTH_MAX];
        while (fgets(temp, CHAR_LENGTH_MAX, In_File) != NULL)
        {
            scanf_ret = fprintf(mdinfo, "    %s", temp);
        }
        scanf_ret = fprintf(mdinfo, "\n\n");
        fclose(In_File);
    }
    
}
void CONTROLLER::Set_Command(const char *Flag, const char *Value, int Check, const char *prefix)
{
    if (prefix && strcmp(prefix, "comments") == 0)
        return;
    char temp[CHAR_LENGTH_MAX] = { 0 };
    if (prefix && prefix[0] != 0 && strcmp(prefix, "main") != 0)
    {
        strcpy(temp, prefix);
        strcat(temp, "_");
    }
    strcat(temp, Flag);
    if (commands.count(temp))
    {
        fprintf(stderr, "\nError: %s is set more than once.\n", temp);
        getchar();
        exit(1);
    }
    original_commands[temp] = Value;
    char temp2[CHAR_LENGTH_MAX];
    sscanf(Value, "%s", temp2);
    commands[temp] = temp2;
        
    command_check[temp] = Check;
}

void CONTROLLER::Default_Set()
{
    srand((unsigned)time(NULL));
}

void CONTROLLER::Init_Cuda()
{
    printf("    Start initializing CUDA\n");
    int count;
    int target = atoi(Command("device"));
    cudaGetDeviceCount(&count);

    printf("        %d device found:\n",count);
    cudaDeviceProp prop;
    float GlobalMem;
    for (int i = 0; i < count; i++)
    {
        cudaGetDeviceProperties(&prop, i);
        GlobalMem = (float) prop.totalGlobalMem / 1024.0f / 1024.0f / 1024.0f;
        printf("            Device %d:\n                Name: %s\n                Memory: %.1f GB\n", i, prop.name, GlobalMem);
    }
    if (count <= target)
    {
        printf("        Error: The available device count %d is less than the setting target %d.", count, target);
        exit(0);
    }
    printf("        Set Device %d\n", target);
    cudaSetDevice(target);
    printf("    End initializing CUDA\n");
}

void CONTROLLER::Input_Check()
{
    if (!(Command_Exist("dont_check_input") && atoi(Command("dont_check_input"))))
    {
        int no_warning = 0;
        for (CheckMap::iterator iter = command_check.begin(); iter != command_check.end(); iter++)
        {
            if (iter->second == 1)
            {
                printf("Warning: '%s' is set, but never used.\n", iter->first.c_str());
                no_warning += 1;
            }
        }
        for (CheckMap::iterator iter = choice_check.begin(); iter != choice_check.end(); iter++)
        {
            if (iter->second == 2)
            {
                printf("Warning: the value '%s' of command '%s' matches none of the choices.\n", this->commands[iter->first].c_str(), iter->first.c_str());
                no_warning += 1;
            }
            else if (iter->second == 3)
            {
                printf("Warning: command '%s' is not set.\n", iter->first.c_str());
                no_warning += 1;
            }
        }
        if (no_warning)
        {
            printf("\nWarning: inputs raised %d warning(s). If You know WHAT YOU ARE DOING, press any key to continue. Set dont_check_input = 1 to disable this warning.\n", no_warning);
            getchar();
        }
    }
}

void CONTROLLER::printf(const char *fmt, ...)
{
    va_list argp;

    va_start(argp, fmt);
    vfprintf(stdout, fmt, argp);
    va_end(argp);

    if (mdinfo != NULL)
    {
        va_start(argp, fmt);
        vfprintf(mdinfo, fmt, argp);
        va_end(argp);
    }
}

void CONTROLLER::Step_Print_Initial(const char *head, const char *format)
{
    outputs_format.insert(std::pair<std::string, std::string>(head, format));
    outputs_content.insert(std::pair<std::string, std::string>(head, "****"));
    outputs_key.push_back(head);
}

void CONTROLLER::Step_Print(const char *head, const float *pointer)
{
    char temp[CHAR_LENGTH_MAX];
    if (outputs_content.count(head))
    {
        sprintf(temp, outputs_format[head].c_str(), pointer[0]);
        outputs_content[head] = temp;
    }
    
}

void CONTROLLER::Step_Print(const char *head, const float pointer)
{
    char temp[CHAR_LENGTH_MAX];
    if (outputs_content.count(head))
    {
        sprintf(temp, outputs_format[head].c_str(), pointer);
        outputs_content[head] = temp;
    }
}

void CONTROLLER::Step_Print(const char *head, const double pointer)
{
    char temp[CHAR_LENGTH_MAX];
    if (outputs_content.count(head))
    {
        sprintf(temp, outputs_format[head].c_str(), pointer);
        outputs_content[head] = temp;
    }
}

void CONTROLLER::Step_Print(const char *head, const int pointer)
{
    char temp[CHAR_LENGTH_MAX];
    if (outputs_content.count(head))
    {
        sprintf(temp, outputs_format[head].c_str(), pointer);
        outputs_content[head] = temp;
    }
}

void CONTROLLER::Print_First_Line_To_Mdout(FILE *mdout)
{
    if (mdout == NULL)
    {
        mdout = this->mdout;
    }
    for (int i = 0; i < outputs_key.size(); i++)
    {
        fprintf(mdout, "%12s ", outputs_key[i].c_str());
    }
    fprintf(mdout, "\n");
    printf("---------------------------------------------------------------------------------------\n");
}

void CONTROLLER::Print_To_Screen_And_Mdout(FILE *mdout)
{
    if (mdout == NULL)
    {
        mdout = this->mdout;
    }
    int line_numbers = 0;
    for (int i = 0; i < outputs_key.size(); i++)
    {
        line_numbers++;
        fprintf(stdout, "%12s = %12s, ", outputs_key[i].c_str(), outputs_content[outputs_key[i]].c_str());
        fprintf(mdout, "%12s ", outputs_content[outputs_key[i]].c_str());
        outputs_content[outputs_key[i]] = "****";
        if (line_numbers % 3 == 0)
            fprintf(stdout, "\n");
        
    }
    if (line_numbers % 3 != 0)
        fprintf(stdout, "\n");
    fprintf(stdout, "---------------------------------------------------------------------------------------\n");
    fprintf(mdout, "\n");
}

void CONTROLLER::Initial(int argc, char **argv)
{
    Arguments_Parse(argc, argv);
    Commands_From_In_File(argc, argv);
    printf("START INITIALIZING CONTROLLER\n");
    Default_Set();
    Init_Cuda();
    is_initialized = 1;
    if (is_initialized && !is_controller_printf_initialized)
    {
        is_controller_printf_initialized = 1;
        printf("    structure last modify date is %d\n", last_modify_date);
    }
    printf("END INITIALIZING CONTROLLER\n\n");
}

void CONTROLLER::Clear()
{
    if (is_initialized)
    {
        is_initialized = 0;

        original_commands.clear();
        commands.clear();
        command_check.clear();

        fclose(mdinfo);
        fclose(mdout);
        outputs_content.clear();
        outputs_format.clear();
        outputs_key.clear();
    }
}

void TIME_RECORDER::Start()
{
    start_timestamp = clock();
}


void TIME_RECORDER::Stop()
{
    end_timestamp = clock();
    time += (double)(end_timestamp - start_timestamp) / CLOCKS_PER_SEC;
}

void TIME_RECORDER::Clear()
{
    time = 0;
    start_timestamp = 0;
    end_timestamp = 0;
}
