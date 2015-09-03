#!/usr/bin/env sh

for f in sparsenet_v.10*; do mv "$f" "${f/v.10/v.90}"; done
