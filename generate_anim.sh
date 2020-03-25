#!/usr/bin/env bash

ffmpeg -y -r 4 -pattern_type glob -i "images/*.png" -pix_fmt yuv420p -r 40 output.mp4