import glob

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDAExtension

sources = glob.glob("src/openpi_cuda/*.cpp") + glob.glob("src/openpi_cuda/*.cu")

setup(
    name="openpi_cuda_lib",
    ext_modules=[
        CUDAExtension(
            name="openpi_cuda_lib",
            sources=sources,
            extra_compile_args={"cxx": ["-g", "-O3"], "nvcc": ["-O3", "--use_fast_math", "-lineinfo"]},
            extra_link_args=["-Wl,--no-as-needed", "-lcuda"],
            include_dirs=["src/openpi_cuda"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
