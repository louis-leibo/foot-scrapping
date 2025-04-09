from setuptools import setup, find_packages

setup(
    name="fbref_scraper",
    version="0.1.0",
    description="Scraper pour collecter les données des joueurs de football depuis FBref",
    author="Manus Agent",
    author_email="agent@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "soccerdata",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "jupyter",
    ],
    python_requires=">=3.8",
)
