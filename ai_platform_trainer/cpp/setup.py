from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import setuptools

# Based on the PyBind11 example setup
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            import subprocess
            subprocess.check_call(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extension")
            
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(
    name='gpu_environment',
    version='0.1.0',
    author='AI Platform Trainer Team',
    description='GPU-accelerated game environment for reinforcement learning',
    long_description='',
    ext_modules=[CMakeExtension('gpu_environment')],
    cmdclass=dict(build_ext=CMakeBuild),
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.0',
        'stable-baselines3>=1.0.0',
        'gym>=0.17.0',
        'numpy>=1.19.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    zip_safe=False,
)
