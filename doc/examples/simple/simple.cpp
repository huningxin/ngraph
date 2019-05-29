//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <iostream>

#include <ngraph/ngraph.hpp>

using namespace ngraph;

// Build a simple model to compute: (tensor0 + tensor1) * (tensor2 + tensor3).
// The tensor0 and tensor2 are constants. The tensor1 and tensor3 are inputs.

// tensor0 ---+
//            +--- ADD ---> intermediateOutput0 ---+
// tensor1 ---+                                    |
//                                                 +--- MUL---> output
// tensor2 ---+                                    |
//            +--- ADD ---> intermediateOutput1 ---+
// tensor3 ---+

int main()
{
    // Build the graph
    const Shape shape{2, 2, 2, 2};
    const size_t size = shape_size(shape);
    const std::vector<float> constant_data(size, 0.5);

    auto tensor0 = std::make_shared<op::Constant>(element::f32, shape, constant_data);
    auto tensor1 = std::make_shared<op::Parameter>(element::f32, shape);
    auto tensor2 = std::make_shared<op::Constant>(element::f32, shape, constant_data);
    auto tensor3 = std::make_shared<op::Parameter>(element::f32, shape);

    auto add0 = std::make_shared<op::Add>(tensor0, tensor1);
    auto add1 = std::make_shared<op::Add>(tensor2, tensor3);

    auto mul = std::make_shared<op::Multiply>(add0, add1);

    // Make the function for the graph
    // The 1st argument specifies the results/outputs. 
    // The 2nd argument specifies the inputs.
    auto function = std::make_shared<Function>(NodeVector{mul},
                                               ParameterVector{tensor1, tensor3});

    // Create the backend and compile the function
    auto backend = runtime::Backend::create("CPU");
    auto exec = backend->compile(function);

    // Allocate tensors for inputs
    auto input0 = backend->create_tensor(element::f32, shape);
    auto input1 = backend->create_tensor(element::f32, shape);

    // Allocate tensor for output
    auto output = backend->create_tensor(element::f32, shape);

    // Initialize the input tensors
    const std::vector<float> input_data0(size, 1), input_data1(size, 2);
    input0->write(input_data0.data(), 0, sizeof(float)*input_data0.size());
    input1->write(input_data1.data(), 0, sizeof(float)*input_data1.size());

    // Invoke the function
    exec->call({output}, {input0, input1});

    // Get the result
    std::vector<float> output_data(size);
    output->read(output_data.data(), 0, sizeof(float)*output_data.size());

    // Print out the result
    std::cout << "[";
    for (size_t i = 0; i < size; ++i)
    {
        std::cout << output_data[i] << ' ';
    }
    std::cout << ']' << std::endl;

    return 0;
}
