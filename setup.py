from setuptools import setup, find_packages, Extension

setup(
    name="ahoi",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tqdm"
    ],
    ext_modules=[Extension("ahoi_scan", sources=["src/ahoi_scan.c"])],
)
