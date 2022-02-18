#include "Network.hpp"

deep::Layer::Layer(cppm::uint64 const& size, deep::Activator activator):
    _values(size),
    _activated(size)
{
    _activator = deep::ACTIVATORS[activator];
    _dx_activator = deep::DX_ACTIVATORS[activator];
}
deep::Layer::Layer(cppm::uint64 const& size,
              double (*activator)(double const), double (*dx_activator)(double const)):
    _values(size),
    _activated(size)
{
    _activator = activator;
    _dx_activator = dx_activator;
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
