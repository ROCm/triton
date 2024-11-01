from hip import hip

# Copied from https://rocm.docs.amd.com/projects/hip-python/en/latest/user_guide/1_usage.html


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result
