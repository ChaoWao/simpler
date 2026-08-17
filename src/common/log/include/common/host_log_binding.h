/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#pragma once

#include <dlfcn.h>

#include "common/host_log_state.h"

namespace simpler::log {

inline int bind_loaded_host_log_state(void *handle, SimplerHostLogState *state, const char **error) {
    if (error != nullptr) *error = nullptr;
    if (handle == nullptr) {
        if (error != nullptr) *error = "module handle is null";
        return -1;
    }
    if (state == nullptr) {
        if (error != nullptr) *error = "host-log state is null";
        return -1;
    }

    dlerror();
    auto bind_state = reinterpret_cast<SimplerHostLogBindStateFn>(dlsym(handle, "simpler_host_log_bind_state"));
    const char *symbol_error = dlerror();
    if (symbol_error != nullptr) {
        if (error != nullptr) *error = symbol_error;
        return -1;
    }
    if (bind_state == nullptr) {
        if (error != nullptr) *error = "host-log state binder symbol is null";
        return -1;
    }
    if (bind_state(state) != 0) {
        if (error != nullptr) *error = "module rejected the host-log state ABI";
        return -1;
    }
    return 0;
}

}  // namespace simpler::log
