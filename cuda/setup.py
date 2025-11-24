"""
tell Python how to compile the C++ and CUDA code into a Python module that can be imported
"""

# for defining a package, extensions, and running build/install logic
from setuptools import setup 
# so pytorch can read it
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='fused_bias_gelu_ext',
    ext_modules=[
        CUDAExtension( # "this extension is C++ and CUDA code"
            name='fused_bias_gelu_ext',
            sources=[
                'bias_gelu_ext.cpp',
                'bias_gelu_kernel.cu',
            ],
            extra_compile_args={
                # high optimization flags for both 
                # necessary for GPU optimization 
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension} # for handling specific CUDA compilation 
)
