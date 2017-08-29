// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

//================================================================================================
// ElementType
//================================================================================================

#pragma once

#include <map>
#include <string>

namespace ngraph
{
    namespace element
    {
        class Type
        {
        public:
            Type(size_t bitwidth, bool is_float, bool is_signed, const std::string& cname);

            const std::string& c_type_string() const;
            size_t             size() const;
            size_t             hash() const
            {
                std::hash<std::string> h;
                return h(m_cname);
            }

            bool operator==(const Type& other) const;
            bool operator!=(const Type& other) const { return !(*this == other); }

        private:
            static std::map<std::string, Type> m_element_list;
            size_t                             m_bitwidth;
            bool                               m_is_float;
            bool                               m_is_signed;
            const std::string                  m_cname;
        };

        const Type float32_t = Type(32, true, true, "float");
        const Type int8_t    = Type(8, false, true, "int8_t");
        const Type int32_t   = Type(32, false, true, "int32_t");
        const Type int64_t   = Type(64, false, true, "int64_t");
        const Type uint8_t   = Type(8, false, false, "int8_t");
        const Type uint32_t  = Type(32, false, false, "int32_t");
        const Type uint64_t  = Type(64, false, false, "int64_t");
    }
}
