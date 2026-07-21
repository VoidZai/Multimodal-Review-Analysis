from setuptools import setup, find_packages

setup(
    name="cragb",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas",
        "pyarrow",
        "numpy",
        "pyyaml",
        "tqdm",
        "matplotlib",
        "langdetect",
    ],
)