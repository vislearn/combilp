#!/bin/sh

TARGET_DIRECTORY=../../SRMP

if [ ! -d "$TARGET_DIRECTORY" ]; then
	git clone https://github.com/opengm/SRMP.git "$TARGET_DIRECTORY"
	(cd "$TARGET_DIRECTORY" && git apply) <srmp.patch
else
	echo "Skipping SRMP download, directory already exists"
fi
