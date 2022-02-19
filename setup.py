from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dropit',
    ext_modules=[
        CUDAExtension('backward_ops', [
            'layers/csrc/backward_ops.cpp',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)