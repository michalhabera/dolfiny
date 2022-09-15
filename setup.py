from setuptools import setup

setup(name='dolfiny',
      version='0.1',
      description='Dolfin-y high level wrappers for dolfin-x',
      author='Michal Habera',
      author_email='michal.habera@gmail.com',
      packages=['dolfiny'],
      zip_safe=False,
      package_data={'dolfiny': ['localsolver.h']},
      include_package_data=True)
