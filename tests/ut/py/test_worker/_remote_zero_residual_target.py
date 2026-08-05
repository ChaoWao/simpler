# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Import target used to hold a Remote L3 runner inside its startup budget."""

import os
import time

_delay_s = float(os.environ.get("SIMPLER_TEST_REMOTE_IMPORT_DELAY_S", "0"))
if _delay_s > 0:
    time.sleep(_delay_s)


def noop(_orch, _args, _cfg) -> None:
    pass
