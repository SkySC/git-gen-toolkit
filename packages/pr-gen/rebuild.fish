#!/usr/bin/env fish

rm -rf build/ dist/ *.egg-info/

pip install build
mise exec -- python -m build
pip install dist/*.whl --force-reinstall