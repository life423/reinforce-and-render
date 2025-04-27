from setuptools import setup, find_packages

setup(
    name="ai_platform_trainer",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pygame>=2.1",
        "torch>=2.0",
    ],
    entry_points={
        "console_scripts": [
            "ai-trainer = ai_platform_trainer.__main__:main",
        ],
    },
)
