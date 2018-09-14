from setuptools import setup, find_packages

setup(
        name = 'econsieve',
        version = 'alpha',
        author='Gregor Boehl',
        author_email='admin@gregorboehl.com',
        description='Various insanly helpful functions',
        packages = find_packages(),
        install_requires=[
            'sympy',
            'matplotlib',
            'scipy',
            'numpy',
         ],
   )
