import sys
import sysconfig
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

PREFIX = next(
    (root for root in (Path(sys.prefix), Path(sysconfig.get_paths()["purelib"]) / "cmeel.prefix")
     if (root / "include/hpp/fcl/distance.h").is_file()),
    Path(sys.prefix),
)

setup(
    ext_modules=[
        Pybind11Extension(
            "_hexfellow_collision_kernel",
            ["src/hexfellow_deploy/robotics/control/collision_kernel.cpp"],
            cxx_std=17,
            include_dirs=[str(PREFIX / "include"), str(PREFIX / "include/eigen3"), "/usr/include/eigen3"],
            library_dirs=[str(PREFIX / "lib")],
            libraries=["hpp-fcl"],
            runtime_library_dirs=[str(PREFIX / "lib")],
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": build_ext},
)
