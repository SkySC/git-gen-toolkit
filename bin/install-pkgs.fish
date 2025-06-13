#!/usr/bin/env fish

for pkg in packages/*
    if test -f "$pkg/pyproject.toml"
        pip install -e $pkg
    end
end
