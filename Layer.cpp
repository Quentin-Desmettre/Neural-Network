#include "Network.hpp"

deep::Layer::Layer(cppm::uint64 const& size, deep::Activator activator):
    _dx_active(size),
    _active(size)
{
    _activator = deep::ACTIVATORS[activator];
    _dx_activator = deep::DX_ACTIVATORS[activator];
}
deep::Layer::Layer(cppm::uint64 const& size,
              double (*activator)(double const), double (*dx_activator)(double const)):
    _dx_active(size),
    _active(size)
{
    _activator = activator;
    _dx_activator = dx_activator;
}
deep::Layer::Layer(cppm::Matrix<double> const& total, Activator activator)
{
    _dx_active = total;
    _active = total;
    _activator = deep::ACTIVATORS[activator];
    _dx_activator = deep::DX_ACTIVATORS[activator];
}

void deep::Layer::setActivator(Activator activator)
{
    _activator = deep::ACTIVATORS[activator];
    _dx_activator = deep::DX_ACTIVATORS[activator];
}
void deep::Layer::setActivator(double (*activator)(double const), double (*dx_activator)(double const))
{
    _activator = activator;
    _dx_activator = dx_activator;
}

double (*deep::Layer::getActivator(void))(double const)
{
    return _activator;
}
double (*deep::Layer::get_DxActivator(void))(double const)
{
    return _dx_activator;
}

const cppm::Matrix<double>& deep::Layer::get_dxActivated(void) const
{
    return _dx_active;
}
const cppm::Matrix<double>& deep::Layer::get_Activated(void) const
{
    return _active;
}

void deep::Layer::set_dxActivated(const cppm::Matrix<double>& other)
{
    _dx_active = other;
}
void deep::Layer::set_Activated(const cppm::Matrix<double>& other)
{
    _active = other;
}

void deep::Layer::setFrom_WB(cppm::Matrix<double> const& weights, const Layer& prevLay, cppm::Matrix<double> const& biases)
{
    cppm::Matrix<double> tmp = weights * prevLay._active;

    _dx_active = tmp.addAndApply(biases, _dx_activator);
    _active = tmp.addAndApply(biases, _activator);
}
