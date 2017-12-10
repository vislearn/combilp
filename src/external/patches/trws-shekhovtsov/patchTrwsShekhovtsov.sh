#!/bin/bash
set -eu -o pipefail

ARCHIVE_FILE=../zip_files/trws_shekhovtsov.tar.gz
TARGET_DIRECTORY=../../trws-shekhovtsov.src-patched

if [[ -e "$TARGET_DIRECTORY" ]]; then
	echo "Source folder already exists, skipping patch."
	exit 0
fi

echo "Downloading TRWS_SHEKHOVTSOV..."
if [[ ! -e "$ARCHIVE_FILE" ]]; then
	wget -qO "$ARCHIVE_FILE" 'https://gitlab.icg.tugraz.at/shekhovt/part_opt/repository/archive.tar.gz?ref=67c45b2dc6d96fe8d8aaa430d50e4bf55823d8a0'
fi

echo "Extracting files for TRWS_SHEKHOVTSOV..."
mkdir -p "$TARGET_DIRECTORY"
tar --strip-components=1 -C "$TARGET_DIRECTORY" -xzf "$ARCHIVE_FILE"

# Remove all unnecessary files from target directory.
# Patch the sources to use a dedicated top-level include directory (this
# reduces chance of name clashes due to very generic include paths).
# Additionally, install a short-hand header to usage easier.
pushd "$TARGET_DIRECTORY" >/dev/null
ls | grep -v code | xargs rm -rf
mv code trws_shekhovtsov
find -type f | xargs -d'\n' perl -pi -e 's/\r\n/\n/g;' -e 's!#include ["<]((?:data/|debug/|defs\.h|dynamic/|exttype/|files/|geom/|massert\.h|maxflow/|mex/|optim/|qpbo-v1.3/|streams/).*)[">]!#include <trws_shekhovtsov/\1>!g;' --
echo '#include <trws_shekhovtsov/optim/part_opt/trws_machine.h>' >trws_shekhovtsov/trws_machine.h
popd >/dev/null

echo "Patching TRWS_SHEKHOVTSOV..."
for i in fix-build.diff redistribute-potentials.diff; do
	patch -d "$TARGET_DIRECTORY" -p 1 <"$i"
done

echo "Done."
