# Neural Network

This project is my first attempt at making a machine learning library.
My goal is to make it able to create an AI, train it over fixed examples, and later use it again to solve new problem.
It is still in development.

## Installation

To use it, you currently just need to clone the repository.

```bash
git clone https://github.com/Quentin-Desmettre/neural-network.git
```

## Usage

```cpp
// To use the library
#include "neural-network/network.hpp"

int main(void)
{
    // create a neural network with 5 input neurons, 2 hidden layers of 16 neurons, and 2 output neurons
    deep::Network net(5, 16, 16, 2);
    
    // get the training examples and their expected result
    std::vector<cppm::Matrix<double>> input_examples = getInputExamples();
    std::vector<cppm::Matrix<double>> example_outputs = getOutputExamples();

    // train the network over examples
    net.train<double>(input_examples, output_examples);
    
    // Save the network for future use
    net.saveToFile("net.nw");
    
    // Load the network from a file
    deep::Network other("net.nw");
    
    // Use your network!
    cppm::Matrix<double> new_input = getInput();
    cppm::Matrix<double> result = other.predict<double>(new_input); // return a column matrix of size 2, as the network has 2 output neurons
    
    return 0;
}
```
