from setuptools import setup

setup(
    name='ScatTools',
    version='0.1',
    author='Timothy Lang',
    author_email='timothy.j.lang@nasa.gov',
    packages=['scattools', ],
    license='',
    description='Scatterometer Python Library',
    long_description=open('description.txt').read(),
    install_requires=['numpy', 'netCDF4', 'pydap', 'scipy'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console"
        ],
)
