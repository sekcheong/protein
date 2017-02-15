#!/bin/bash
#
#This script runs a test for Assignment 3, part 2.

files=""

nhid=2
for learn in and or xor same ; do
  for lr in 0.02 0.1 ; do
    for mom in 0.7 ; do
      for hidlayers in 1 ; do
        f=$learn-$lr-$mom-$hidlayers-$nhid.out
        java TesterPart2 $learn.data 1000 $lr $mom 0.1 $hidlayers $nhid > $f
        files="$files $f"
      done
    done
  done
done

learn=auto8
hidlayers=1
nhid=3
for lr in 0.02 ; do
  for mom in 0.0 0.7 ; do
    f=$learn-$lr-$mom-$hidlayers-$nhid.out
    java TesterPart2 $learn.data 1000 $lr $mom 0.1 $hidlayers $nhid > $f
    files="$files $f"
  done
done

learn=multi
lr=0.1
for mom in 0.0 0.7 ; do
  for hidlayers in 1 2 ; do
    for nhid in 2 3 ; do
      f=$learn-$lr-$mom-$hidlayers-$nhid.out
      java TesterPart2 $learn.data 1000 $lr $mom 0.1 $hidlayers $nhid > $f
      files="$files $f"
    done
  done
done

tar -cf /dev/stdout $files | gzip > output2.tgz 
