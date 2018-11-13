//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#pragma once

#include <memory>

#include "ngraph/function.hpp"
#include "ngraph/runtime/performance_counter.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        class ExternalFunction;
        class Tensor;
        class Backend;
        using Handle = void*;
    }
}

/// \brief Interface to a generic backend.
///
/// Backends are responsible for function execution and value allocation.
class ngraph::runtime::Backend
{
public:
    virtual ~Backend();
    /// \brief Create a new Backend object
    /// \param type The name of a registered backend, such as "CPU" or "GPU".
    ///   To select a subdevice use "GPU:N" where s`N` is the subdevice number.
    /// \returns unique_ptr to a new Backend or nullptr if the named backend
    ///   does not exist.
    static std::unique_ptr<Backend> create(const std::string& type);

    /// \brief Query the list of registered devices
    /// \returns A vector of all registered devices.
    static std::vector<std::string> get_registered_devices();

    /// \brief Create a tensor specific to this backend
    /// \param element_type The type of the tensor element
    /// \param shape The shape of the tensor
    /// \returns shared_ptr to a new backend-specific tensor
    virtual std::shared_ptr<ngraph::runtime::Tensor>
        create_tensor(const ngraph::element::Type& element_type, const Shape& shape) = 0;

    /// \brief Create a tensor specific to this backend
    /// \param element_type The type of the tensor element
    /// \param shape The shape of the tensor
    /// \param memory_pointer A pointer to a buffer used for this tensor. The size of the buffer
    ///     must be sufficient to contain the tensor. The lifetime of the buffer is the
    ///     responsibility of the caller.
    /// \returns shared_ptr to a new backend-specific tensor
    virtual std::shared_ptr<ngraph::runtime::Tensor> create_tensor(
        const ngraph::element::Type& element_type, const Shape& shape, void* memory_pointer) = 0;

    /// \brief Create a tensor of C type T specific to this backend
    /// \param shape The shape of the tensor
    /// \returns shared_ptr to a new backend specific tensor
    template <typename T>
    std::shared_ptr<ngraph::runtime::Tensor> create_tensor(const Shape& shape)
    {
        return create_tensor(element::from<T>(), shape);
    }

    /// \brief Compiles a Function.
    /// \param func The function to compile
    /// \returns true if compile is successful, false otherwise
    virtual Handle compile(const std::shared_ptr<Function>& func) = 0;

    /// \brief Executes a single iteration of a Function. If func is not compiled the call will
    ///     compile it.
    /// \param handle Handle returned from compile() or load()
    /// \returns true if iteration is successful, false otherwise
    virtual bool call(Handle handle,
                      const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                      const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) = 0;

    /// \brief Executes a single iteration of a Function. If func is not compiled the call will
    ///     compile it. Optionally validates the inputs and outputs against the function graph.
    /// \param handle Handle returned from compile() or load()
    /// \returns true if iteration is successful, false otherwise
    bool call_with_validate(Handle handle,
                            const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                            const std::vector<std::shared_ptr<runtime::Tensor>>& inputs)
    {
        validate_call(handle, outputs, inputs);
        return call(handle, outputs, inputs);
    }

    /// \brief Compiled functions may be cached. This function removes a compiled function
    ///     from the cache.
    /// \param handle Handle returned from compile() or load()
    virtual void remove_compiled_function(Handle handle);

    /// \brief Save the function's state.
    /// \param handle Handle returned from compile() or load()
    /// \param path File system path to the output file.
    /// \returns true if save is successful.
    virtual bool save(Handle handle, const std::string& path) const;

    /// \brief Load a function's saved state.
    /// \param path File system path to the input file.
    /// \returns non-nullptr Handle if successful, nullptr otherwise.
    virtual Handle load(const std::string& path);

    /// \brief Enable the collection of per-op performance information on a specified Function.
    ///     Data collection is via the `get_performance_data` method.
    /// \param enable Set to true to enable or false to disable data collection
    virtual void enable_performance_data(bool enable);

    /// \brief Collect performance information gathered on a Function.
    /// \param handle Handle returned from compile() or load()
    /// \returns Vector of PerformanceCounter information.
    virtual std::vector<PerformanceCounter> get_performance_data(Handle handle) const;

    /// \brief Test if a backend is capable of supporting an op
    /// \param node is the op to test.
    /// \returns true if the op is supported, false otherwise.
    virtual bool is_supported(const Node& node) const;

    /// \brief Query the input Parameters for a given Handle
    /// \param handle Handle returned from compile() or load()
    /// \returns an ngraph::op::ParameterVector of all input parameters
    virtual const ngraph::op::ParameterVector& get_parameter_descriptors(Handle handle) const = 0;

    /// \brief Query the output Results for a given Handle
    /// \param handle Handle returned from compile() or load()
    /// \returns an ngraph::ResultVector of all input parameters
    virtual const ngraph::ResultVector& get_result_descriptors(Handle handle) const = 0;

protected:
    void validate_call(Handle handle,
                       const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                       const std::vector<std::shared_ptr<runtime::Tensor>>& inputs);
};
