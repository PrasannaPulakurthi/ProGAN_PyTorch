#!/bin/bash

for layer in 'initial.3.conv' 'prog_blocks.0.conv1.conv' 'prog_blocks.0.conv2.conv'  'prog_blocks.1.conv1.conv' 'prog_blocks.1.conv2.conv' 'prog_blocks.2.conv1.conv' 'prog_blocks.2.conv2.conv' 'prog_blocks.3.conv1.conv' 'prog_blocks.3.conv2.conv' 'prog_blocks.4.conv1.conv' 'prog_blocks.4.conv2.conv' 'prog_blocks.5.conv1.conv' 'prog_blocks.5.conv2.conv' 'prog_blocks.6.conv1.conv' 'prog_blocks.6.conv2.conv' 'prog_blocks.7.conv1.conv' 'prog_blocks.7.conv2.conv'  'rgb_layers.1.conv1.conv' 'rgb_layers.2.conv1.conv';
do
    echo $layer
    save_dir="Results/Error_vs_Rank/${layer}"
    echo $save_dir
    sbatch --output=Results/%x_%j.out --error=Results/%x_%j.err --time=0-0:20:0 --job-name="rank" --export=layer=$layer,save_dir=$save_dir TestVsRanks.sh 
done