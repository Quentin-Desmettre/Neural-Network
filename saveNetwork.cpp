#include "Network.hpp"
#include <fstream>
#include <iostream>

void write(std::ofstream &f, const void *data, std::streamsize size)
{
    f.write((const char *)data, size);

    if (!f.good())
        throw "ERROR: saveToFile: Error while writing to buffer.";
}

void encodeMatrix(cppm::Matrix<double> const& m, std::ofstream &f)
{
    const cppm::uint64 *size = m.getSize();

    // 8 bytes: nbr of row
    // 8 bytes: nbr of col
    write(f, size, 8);
    write(f, size + 1, 8);

    // write each elem
    cppm::uint64 const max = size[2];
    const double *elm = m.getElems();
    for (cppm::uint64 i = 0; i < max; i++)
        write(f, elm + i, sizeof(double));
}

int deep::Network::saveToFile(std::string const& file)
{
    /*
    file format:
    8 bytes: number of layers
    for each layer:
        8 bytes: size of current layer
        if i_layer > 0:
            encodeMatrix of weight
            encodeMatrix of bias
    */
    std::ofstream f(file, std::ios::out | std::ios::binary);
    if (!f.is_open())
        throw ("ERROR: saveToFile: Cannot open " + file + ".").c_str();

    write(f, &_nbLayer, 8);

        unsigned char c = deep::SIGMOID;
    for (cppm::uint64 i = 0; i < _nbLayer; i++) {
        write(f, &_sizes[i], 8);
        write(f, &c, 1);
        if (!i)
            continue;
        encodeMatrix(_weights[i - 1], f);
        encodeMatrix(_biases[i - 1], f);
    }
    return 0;
}
