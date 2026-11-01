import os
from glob import glob
from setuptools import find_packages, setup

package_name = "athome_face"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "web"), glob("web/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="cyan",
    maintainer_email="cyan@barelang.org",
    description="Standalone 7-inch HDMI LCD Emotion & Expression Interface for Barelang Inari Robot",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "face_node = athome_face.face_node:main",
        ],
    },
)

