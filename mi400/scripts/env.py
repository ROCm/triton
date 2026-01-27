import os


def getTritonBasePath():
    triton_home = os.environ.get('TRITON_TEST_OUTPUT_DIR')
    if triton_home is None:
        raise Exception(
            'Env var TRITON_TEST_OUTPUT_DIR not found. Please set it to the root folder of the Triton repository')
    return triton_home
