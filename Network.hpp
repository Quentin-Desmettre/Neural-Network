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

    class Layer
    {
    private:
        double (*_activator)(double const);
        double (*_dx_activator)(double const);
        cppm::Matrix<double> _values;
        cppm::Matrix<double> _activated;

    public:
        Layer(cppm::uint64 const& size, Activator activator = SIGMOID);
        Layer(cppm::uint64 const& size,
              double (*activator)(double const), double (*dx_activator)(double const));

        void setActivator(Activator activator);
        void setActivator(double (*activator)(double const), double (*dx_activator)(double const));
        double (*getActivator(void))(double const) const;
        double (*get_DxActivator(void))(double const) const;

        const cppm::Matrix<double>& get_dxActivated(void) const;
        const cppm::Matrix<double>& get_Activated(void) const;

        void set_dxActivated(const cppm::Matrix<double>& other);
        void set_Activated(const cppm::Matrix<double>& other);
    };
};

#endif // AA1E3F5A_8263_4B41_8A6D_9B1C7B5EBE91
