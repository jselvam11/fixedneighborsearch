from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fixedneighborsearch',
    ext_modules=[
        CUDAExtension(
            name='fixedneighborsearch',
            sources=['src/your_extension.cpp', 'src/your_kernel.cu'],  # Your source files
            include_dirs=[
                'src/'
                '/path/to/cub',  # Path to CUB
                '/path/to/fmt/include',  # Path to fmt
            ],
            # PyTorch determines the necessary include paths and libraries for torch itself
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True),
    }
)