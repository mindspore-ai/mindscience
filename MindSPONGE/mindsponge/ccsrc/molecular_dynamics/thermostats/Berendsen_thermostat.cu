#include "Berendsen_thermostat.cuh"



void BERENDSEN_THERMOSTAT_INFORMATION::Initial(CONTROLLER *controller, float target_temperature, const char *module_name)
{
    controller->printf("START INITIALIZING BERENDSEN THERMOSTAT:\n");
    if (module_name == NULL)
    {
        strcpy(this->module_name, "berendsen_thermostat");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    controller->printf("    The target temperature is %.2f K\n", target_temperature);

    this->target_temperature = target_temperature;

    dt = 1e-3f;
    if (controller[0].Command_Exist("dt"))
        dt = atof(controller[0].Command("dt"));
    controller->printf("    The dt is %f ps\n", dt);

    tauT = 1.0f;
    if (controller[0].Command_Exist(this->module_name, "tau"))
        tauT = atof(controller[0].Command(this->module_name, "tau"));
    controller->printf("    The time constant tau is %f ps\n", tauT);

    stochastic_term = 0;
    if (controller[0].Command_Exist(this->module_name, "stochastic_term"))
        stochastic_term = atof(controller[0].Command(this->module_name, "stochastic_term"));
    controller->printf("    The stochastic term is %d\n", stochastic_term);

    if (stochastic_term)
    {
        int seed = time(NULL);
        if (controller[0].Command_Exist(this->module_name, "seed"))
        {
            seed = atoi(controller[0].Command(this->module_name, "seed"));
        }
        controller->printf("    The random seed is %d\n", seed);
        e.seed(seed);
        std::normal_distribution<float> temp(0, dt * CONSTANT_TIME_CONVERTION);
        n = temp;
    }

    is_initialized = 1;
    if (is_initialized && !is_controller_printf_initialized)
    {
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }

    controller->printf("END INITIALIZING BERENDSEN THERMOSTAT\n\n");
}


void BERENDSEN_THERMOSTAT_INFORMATION::Record_Temperature(float temperature, int freedom)
{
    if (is_initialized)
    {
        if (stochastic_term)
        {
            lambda = sqrtf(1 + dt / tauT * (target_temperature / temperature - 1) + 2 * sqrtf(target_temperature / temperature / freedom / tauT) * n(e));
        }
        else
        {    
            lambda = sqrtf(1 + dt / tauT * (target_temperature / temperature - 1));

        }
    }
}

void BERENDSEN_THERMOSTAT_INFORMATION::Scale_Velocity(int atom_numbers, VECTOR *vel)
{
    if (is_initialized)
    {
        Scale_List((float*)vel, lambda, 3 * atom_numbers);
    }
}
