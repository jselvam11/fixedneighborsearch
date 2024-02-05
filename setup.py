from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(
    name='cuda_extension',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
                name='fixed_neighbor_search',
                sources=[
                        'src/BuildSpatialHashTableOpKernel.cu',
                        'src/Dtype.cpp',
                        'src/FixedRadiusSearchOpKernel.cu',
                        'src/FixedRadiusSearchOps.cpp',
                        'src/Helper.cpp',
                        'src/Logging.cpp',
                ],
                 include_dirs=[
                      'src/'
                      '/cub-1.16.0/cub',
                ],
                extra_compile_args={'cxx': ['-DBUILD_CUDA_MODULE'],
                                    'nvcc': ['-O2']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })