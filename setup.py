from setuptools import setup, find_packages
setup(
    name="semrec",
    version="0.2.0",
    description="SemRec: Verification Oracle for Cross-Language Equivalence",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.9",
    install_requires=["z3-solver>=4.12", "openai>=1.0"],
    extras_require={"dev": ["pytest", "pytest-cov"]},
    entry_points={"console_scripts": [
        "semrec=src.semrec_cli:main",
        "cross-lang-verify=src.ub_oracle.cli:main",
    ]},
)
