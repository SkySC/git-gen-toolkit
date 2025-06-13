#!/usr/bin/env fish

# * This script builds and installs the specified package(s) from the git-gen-toolkit.
# * If no package is specified, it will build and install all packages.

set packages "toolkit-utils" "llm-connection" "commit-gen" "pr-gen"
set original_dir (pwd)

if set -q argv[1]
    # * If a specific package is provided, only build and install that package
    if contains $argv[1] $packages
        set packages $argv[1]
    else
        echo "Error: Unknown package '$argv[1]'"
        echo "Available packages: $packages"
        exit 1
    end
end

# * Get the absolute path to Python to avoid mise's auto-virtualenv behavior
set python_path (which python)
echo "Using Python: $python_path"

# * Install core dependencies first
echo "Installing core dependencies..."
$python_path -m pip install questionary gitpython requests types-requests --quiet

# * First, let's build all the packages without installing
echo "Building all packages..."
set build_dirs ""

for pkg in $packages
    set pkg_path $original_dir/packages/$pkg
    
    if test -d $pkg_path
        echo "Building $pkg..."
        
        # * Clean up any existing build artifacts directly (without changing directory)
        if test -d $pkg_path/build
            rm -rf $pkg_path/build
        end
        if test -d $pkg_path/dist
            rm -rf $pkg_path/dist
        end
        for egg_info in $pkg_path/*.egg-info
            if test -d $egg_info
                rm -rf $egg_info
            end
        end
        
        # * Create a temporary directory for building
        set build_dir (mktemp -d)
        set -a build_dirs $build_dir
        echo "Using temporary build directory: $build_dir"
        
        # * Copy package contents to temp dir to avoid mise's auto-virtualenv
        cp -r $pkg_path/* $build_dir/
        
        # * Build the package from the temp directory
        cd $build_dir
        
        # * Install build package in the current environment
        $python_path -m pip install build --quiet
        
        # * Build the package
        $python_path -m build
        
        # * Copy dist back to original location for reference
        if test -d dist
            if not test -d $pkg_path/dist
                mkdir -p $pkg_path/dist
            end
            cp dist/*.whl $pkg_path/dist/
            echo "Built wheel for $pkg in $pkg_path/dist/"
        else
            echo "❌ Failed to build $pkg - no wheel file found"
            cd $original_dir
            exit 1
        end
        
        cd $original_dir
    else
        echo "❌ Package directory not found: $pkg_path"
        exit 1
    end
end

# * Create a temporary directory for all wheels
set wheel_dir (mktemp -d)
echo "Collecting all wheels in $wheel_dir"

# * Copy all wheels to the temporary directory
for pkg in $packages
    set pkg_path $original_dir/packages/$pkg
    if test -d $pkg_path/dist
        cp $pkg_path/dist/*.whl $wheel_dir/
    end
end

# * Install all packages from the wheel directory in a single pip command
# * This avoids dependency resolution issues
echo "Installing all packages from wheels..."
$python_path -m pip install --force-reinstall $wheel_dir/*.whl

if test $status -eq 0
    echo "✅ Successfully installed all packages"
else
    echo "❌ Failed to install packages"
    exit 1
end

# * Clean up temporary directories
for build_dir in $build_dirs
    echo "Cleaning up $build_dir"
    rm -rf $build_dir
end

echo "Cleaning up $wheel_dir"
rm -rf $wheel_dir

echo "All packages installed successfully!"
cd $original_dir