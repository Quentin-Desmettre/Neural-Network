#include "Network.hpp"
#include <thread>
#include <initializer_list>

#define ABS(x) ((x) < 0 ? -(x) : (x))

void deep::Network::_feedForward(cppm::Matrix<double> const& input)
{
    _layers[0].set_Activated(input);
    for (cppm::uint64 i = 0; i < _nbLayer - 1; ++i)
        _layers[i + 1].setFrom_WB(_weights[i], _layers[i], _biases[i]);
}

cppm::Matrix<double> deep::Network::predict(cppm::Matrix<double> const& input)
{
    _feedForward(input);
    return _layers[_nbLayer - 1].get_Activated();
}

deep::Network::~Network()
{
    _weights.clear();
    _biases.clear();
    _layers.clear();
}
