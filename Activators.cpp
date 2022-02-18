#include "Network.hpp"
#include <cmath>

double deep::sigmoid(double const x)
{
    return 1.0 / (1.0 + exp(-x));
}
double deep::dx_sigmoid(double const x)
{
    double const sigx = deep::sigmoid(x);
    return sigx * (1 - sigx);
}

double deep::relu(double const x)
{
    return x <= 0 ? 0 : x;
}
double deep::dx_relu(double const x)
{
    //     x < 0 ? 0 : 1;
    return x > 0;
}

double deep::swish(double const x)
{
    return x / (1.0 + exp(-x));
}
double deep::dx_swish(double const x)
{
    double const swishx = deep::swish(x);
    double const sigx = deep::sigmoid(x);
    return swishx + sigx * (1 - swishx);
}

double deep::relu6(double const x)
{
    if (x <= 0)
        return 0;
    if (x >= 6)
        return 6;
    return x;
}
double deep::dx_relu6(double const x)
{
    // if (x <= 0 || x => 6)
    //     return 0;
    // return 1;
    return (x > 0 && x < 6);
}

double deep::hardswish(double const x)
{
    return x * (deep::relu6(x + 3) / 6.0);
}
double deep::dx_hardswish(double const x)
{
    return (1 / 6.0) * (deep::relu6(x + 3) + x * deep::dx_relu6(x + 3));
}

double deep::tanh(double const x)
{
    return tanh(x);
}
double deep::dx_tanh(double const x)
{
    double sech = 1.0 / cosh(x);
    return sech * sech;
}
