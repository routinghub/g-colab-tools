import setuptools

setuptools.setup(
    name="routinghub-g-collab-tools",
    version="0.0.1",
    author="Denis Kurilov",
    author_email="denis@routinghub.com",
    description="Google Collab tools",
    url="https://github.com/routinghub/g-collab-tools",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        'pandas',
        'requests',
        'xlrd'
    ],
)

