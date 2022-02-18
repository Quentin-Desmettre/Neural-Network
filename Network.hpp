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
        cppm::Matrix<double> _dx_active;
        cppm::Matrix<double> _active;

    public:
        Layer(cppm::uint64 const& size, Activator activator = SIGMOID);
        Layer(cppm::uint64 const& size,
              double (*activator)(double const), double (*dx_activator)(double const));

        void setActivator(Activator activator);
        void setActivator(double (*activator)(double const), double (*dx_activator)(double const));
        double (*getActivator(void))(double const);
        double (*get_DxActivator(void))(double const);

        const cppm::Matrix<double>& get_dxActivated(void) const;
        const cppm::Matrix<double>& get_Activated(void) const;

        void set_dxActivated(const cppm::Matrix<double>& other);
        void set_Activated(const cppm::Matrix<double>& other);

        void setFrom_WB(cppm::Matrix<double> const& weights, const Layer& prevLay, cppm::Matrix<double> const& biases);
    };

    class Network
    {
    private:
        std::vector<cppm::Matrix<double>> _weights;
        std::vector<cppm::Matrix<double>> _biases;
        std::vector<Layer> _layers;

        cppm::uint64 _nbLayer;

        template<typename T>
        void _getSize(std::vector<T> &sizes, T a)
        {
            if (!a)
                throw "Invalid layer size";
            sizes.push_back(a);
        }
        template <typename T, typename... Args>
        void _getSize(std::vector<T> &sizes, T a, Args&&... args)
        {
            if (!a)
                throw "Invalid layer size";
            sizes.push_back(a);
            _getSize<T>(sizes, args...);
        }

        void _feedForward(cppm::Matrix<double> const& input);
    public:
        template <typename... Args>
        Network(Args&&... args)
        {
            //srand(time(nullptr));
            std::vector<cppm::uint64> sizes;

            _getSize<cppm::uint64>(sizes, args...);
            _nbLayer = sizes.size();
            if (_nbLayer < 2)
                throw "Invalid network size";
            for (cppm::uint64 i = 0; i < _nbLayer; i++) {
                // create a layer
                _layers.push_back(Layer(sizes[i], SIGMOID));
                if (i) {

                    // create the weights and biases for the next layer
                    _weights.push_back(cppm::Matrix<double>(sizes[i], sizes[i - 1]));
                    _biases.push_back(cppm::Matrix<double>(sizes[i], 1));
                    // init the weights and biases
                    for (cppm::uint64 j = 0, n = sizes[i]; j < n; j++) {
                        _biases.back().at(j, 0) = 1;
                        for (cppm::uint64 k = 0, m = sizes[i - 1]; k < m; k++)
                            _weights.back().at(j, k) = 0.5;
                    }
                }
            }
        }

        const std::vector<Layer> &getLayers(void) const {return _layers;}
        void setLayerActivator(cppm::uint64 index, Activator acti)
        {
            _layers[index].setActivator(acti);
        }
        void setLayerActivator(cppm::uint64 index, double (*activator)(double const), double (*dx_activator)(double const))
        {
            _layers[index].setActivator(activator, dx_activator);
        }

        cppm::Matrix<double> predict(cppm::Matrix<double> const& input);
        void train(inputVector const& data, outputVector const& expect);
        ~Network();
    };

};

#endif // AA1E3F5A_8263_4B41_8A6D_9B1C7B5EBE91
