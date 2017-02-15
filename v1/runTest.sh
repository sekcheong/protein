#!/bin/bash

files=""

for learn in and or xor same ; do
  for lr in 0.02 0.1 0.5 ; do
    for mom in 0.0 0.5 0.9 ; do
      f=$learn-$lr-$mom.out
      java TesterPart1 $learn.data 1000 $lr $mom 0.1 > $f
      files="$files $f"
    done
  done
done

tar -cf /dev/stdout $files | gzip > output.tgz 
