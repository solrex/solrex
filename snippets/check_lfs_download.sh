#!/bin/bash

# Get lfs files sha256 checksum from orignal git repo
# Check lfs files sha256 checksum in seperated downloding directory

WORKDIR=$PWD && \
echo -e "Git Repo Dir: $1\nHFD Download Dir: $WORKDIR" && \
cd $1 && echo "Getting lfs files sha256 checksum from: $1" && \
git lfs ls-files -l | awk '{print $1"  "$3}' > $WORKDIR/lfs_files.sha256 && \
echo -e "Writing checksum to $WORKDIR/lfs_files.sha256\nChecking lfs files sha256 checksum: " && \
cd $WORKDIR && sha256sum -c lfs_files.sha256
