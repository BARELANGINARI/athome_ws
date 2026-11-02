from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'athome_hri'

# Perbaikan bug colcon_core symlink_data pada Ubuntu 24.04 (os.remove dipanggil pada target non-existent)
try:
    from colcon_core.distutils.commands.symlink_data import symlink_data as _symlink_data

    class symlink_data(_symlink_data):
        """Fix bug in colcon_core symlink_data where os.remove(target) is called on non-existent target."""

        def copy_file(self, src, dst, **kwargs):
            if kwargs.get('link'):
                return super().copy_file(src, dst, **kwargs)
            if self.force:
                target = os.path.join(dst, os.path.basename(src)) if os.path.isdir(dst) else dst
                if os.path.islink(target) or os.path.exists(target):
                    os.remove(target)
            kwargs['link'] = 'sym'
            src = os.path.abspath(src)
            return super(_symlink_data, self).copy_file(src, dst, **kwargs)

    cmdclass = {'symlink_data': symlink_data}
except ImportError:
    cmdclass = {}

setup(
    name=package_name,
    version='0.1.0',
    cmdclass=cmdclass,
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
            'help_me_carry_hri = athome_hri.help_me_carry_hri:main_entry',
            'barelang_hri = athome_hri.help_me_carry_hri:main_entry',
            'stt_engine_node = athome_hri.stt_engine_node:main',
            'tts_engine_node = athome_hri.tts_engine_node:main',
        ],
    },
)
