from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'hri'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [f for f in glob('launch/*.launch.py') if os.path.isfile(f)]),
        ('share/' + package_name + '/config', [f for f in glob('config/*') if os.path.isfile(f)]),
        ('share/' + package_name + '/src', [f for f in glob('src/*') if os.path.isfile(f)]),
        ('share/' + package_name + '/model', [f for f in glob('model/*') if os.path.isfile(f)]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yesaya',
    maintainer_email='yesaya@todo.todo',
    description='Barelang Inari Human-Robot Interaction (HRI) Package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'barelang_hri = hri.barelang_hri:main_entry',
        ],
    },
)

