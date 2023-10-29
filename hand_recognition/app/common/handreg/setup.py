#! -*- coding:utf-8 -*- 

from distutils.core import setup, Extension
import os 

example_module = Extension('_handreg',
                           sources=['handReg_wrap.cxx'],
                           library_dirs=['./'], 
                           libraries=['handReg'], 
                           )

setup (name = 'handreg',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig example from docs""",
       ext_modules = [example_module],
       py_modules = ["handreg"],
       )

os.system('cp libhandReg.so /usr/lib/')