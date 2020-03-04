from setuptools import setup, find_packages, Extension

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="ahoi",
    version="0.4",
    packages=["ahoi"],
    install_requires=["numpy", "tqdm"],
    python_requires=">2.7",
    ext_modules=[Extension("ahoi.ahoi_scan", sources=["src/ahoi_scan.c"])],
    author="Nikolai Hartmann",
    author_email="nikoladze@posteo.de",
    description="Brute-force scan for rectangular cuts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/nikoladze/ahoi",
)
