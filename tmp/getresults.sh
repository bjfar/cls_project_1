#!/bin/bash

# Just a quick script to dig results out of the directory tree and collect them here

find ../results/0lep -name "*-XSxBR_limits.txt" | while read oldname; do cp "$oldname" 0lep-"${oldname##*/}"; done
find ../results/1lep -name "*-XSxBR_limits.txt" | while read oldname; do cp "$oldname" 1lep-"${oldname##*/}"; done
find ../results/2b -name "*-XSxBR_limits.txt"   | while read oldname; do cp "$oldname" 2b-"${oldname##*/}"; done
find ../results/2lep -name "*-XSxBR_limits.txt" | while read oldname; do cp "$oldname" 2lep-"${oldname##*/}"; done
