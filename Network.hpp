#ifndef AA1E3F5A_8263_4B41_8A6D_9B1C7B5EBE91
#define AA1E3F5A_8263_4B41_8A6D_9B1C7B5EBE91

#include "Matrix.hpp"
#include <exception>
#include <cstdlib>
#include <ctime>

#define randFloat() ((double) rand() / (RAND_MAX))

typedef std::vector<cppm::Matrix<double>> inputVector;
typedef std::vector<cppm::Matrix<double>> outputVector;

namespace deep {

    enum Activator{SIGMOID, RELU, SWISH, TANH, RELU6, HARDSWISH};

    double sigmoid(double const x);
    double dx_sigmoid(double const x);

    double relu(double const x);
    double dx_relu(double const x);

    double swish(double const x);
    double dx_swish(double const x);

    double tanh(double const x);
    double dx_tanh(double const x);

    double relu6(double const x);
    double dx_relu6(double const x);

    double hardswish(double const x);
    double dx_hardswish(double const x);

    static double (*ACTIVATORS[])(double const) = {&sigmoid, &relu, &swish, &tanh, &relu6, &hardswish};
    static double (*DX_ACTIVATORS[])(double const) = {&dx_sigmoid, &dx_relu, &dx_swish, &dx_tanh, &dx_relu6, &dx_hardswish};

};

#endif // AA1E3F5A_8263_4B41_8A6D_9B1C7B5EBE91
