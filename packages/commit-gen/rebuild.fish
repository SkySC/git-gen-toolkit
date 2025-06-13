#!/usr/bin/env fish

# * Clean up any existing build artifacts
rm -rf build dist
set egg_info_files *.egg-info
if test -n "$egg_info_files[1]"
    rm -rf $egg_info_files
end

# * Install build package
python -m pip install build --quiet

# * Build the package
python -m build

# * Install the wheel
python -m pip install dist/*.whl --force-reinstall --no-deps