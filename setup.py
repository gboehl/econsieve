from setuptools import setup, find_packages

setup(
        name = 'econsieve',
        version = '0.0.1',
        author='Gregor Boehl',
        author_email='admin@gregorboehl.com',
        description='linear and nonlinear Bayesian filters',
        packages = find_packages(),
        install_requires=[
            'sympy',
            'matplotlib',
            'scipy',
            'numpy',
            'numba',
         ],
   )
