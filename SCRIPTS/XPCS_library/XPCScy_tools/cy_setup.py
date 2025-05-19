import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


ext_modules = [
    Extension(
        "XPCScy_tools",
        ["XPCScy_tools.pyx"],
        libraries=["m"],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
        extra_link_args=['-fopenmp'],
    )
]


setup(
  name="XPCScy_tools",
  cmdclass={"build_ext": build_ext},
  ext_modules=ext_modules
)


# compile instructions:
# python cy_setup.py build_ext --inplace