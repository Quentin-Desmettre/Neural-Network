#include "Network.hpp"
#include <fstream>
#include <iostream>

void read(std::ifstream &f, void *data, std::streamsize size)
{
    f.read((char *)data, size);
    if (!f.good())
        throw "ERROR: createFromFile: Error while reading from buffer.";
}

cppm::Matrix<double> decodeMatrix(std::ifstream &f)
{
    // 8 bytes: nbr of row
    // 8 bytes: nbr of col
    cppm::uint64 nRow, nCol;

    read(f, &nRow, 8);
    read(f, &nCol, 8);

    cppm::Matrix<double> m(nRow, nCol);
    double tmp;

    // read each elem
    for (cppm::uint64 i = 0; i < nRow; i++)
        for (cppm::uint64 j = 0; j < nCol; j++) {
            read(f, &tmp, 8);
            m.at(i, j) = tmp;
        }
    return m;
}

void deep::Network::_initFromFile(const std::string &file)
{
    /*
    file format:
    8 bytes: number of layers
    for each layer:
        8 bytes: size of current layer
        1 byte: activation function
        if i_layer > 0:
            decodeMatrix of weight
            decodeMatrix of bias
    */
    std::ifstream f(file, std::ios::in | std::ios::binary);

    if (!f.is_open())
        throw ("ERROR: createFromFile: Cannot open " + file + ".").c_str();

    // get number of layer
    cppm::uint64 size;
    read(f, &size, 8);
    _nbLayer = size;

    for (cppm::uint64 i = 0; i < size; i++) {
        // get layer size and activation function
        cppm::uint64 laySize;
        unsigned char act;
        read(f, &laySize, 8);
        read(f, &act, 1);

        if (act > deep::NB_ACTIVATOR)
            throw "ERROR: createFromFile: Invalid activation function.";

        _sizes.push_back(laySize);
        _layers.push_back(Layer(laySize, deep::Activator(act)));
        if (!i)
            continue;
        // get weights and biases
        _weights.push_back(decodeMatrix(f));
        _biases.push_back(decodeMatrix(f));
    }
}
