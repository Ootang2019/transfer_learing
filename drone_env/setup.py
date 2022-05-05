from setuptools import find_packages, setup

with open("README.rst", "r") as longdesc:
    long_description = longdesc.read()

setup(
    name="drone-env",
    description="a drone environment with different tasks",
    long_description=long_description,
    author="Yu Tang Liu",
    version="0.0.1",
    packages=find_packages(where="drone_env/"),
    package_dir={"": "drone_env"},
    install_requires=["gym==0.18.0"],
)
