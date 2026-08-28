from setuptools import setup
from Cython.Build import cythonize
setup(ext_modules=cythonize("sweep_cy.pyx", language_level=3,
                            compiler_directives={"boundscheck": False,
                                                 "wraparound": False}))
