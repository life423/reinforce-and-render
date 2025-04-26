from setuptools import find_packages, setup

setup(
    name="ai_platform_trainer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pygame>=2.5.0",
        "numpy>=1.20.0",
        # Add all other dependencies here
    ],
    entry_points={
        "console_scripts": [
            # "run-game" is the command name you’ll type in Terminal;
            # "ai_platform_trainer.main:main" is the Python function to run.
            "run-game = ai_platform_trainer.main:main"
        ]
    },
)
