# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import subprocess

def rocm_get_gpuarch():
    lines = rocm_get_allarch()
    assert lines
    # Example: gfx942:sramecc+:xnack-
    return lines[0].split(':')[0]

def rocm_get_allarch():
    out = subprocess.check_output(['rocm_agent_enumerator -name'], shell=True).decode('utf8', errors='ignore').strip()
    return [line for line in out.splitlines() if not 'generic' in line ]

if __name__ == '__main__':
    print(rocm_get_allarch())
