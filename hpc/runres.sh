#!/bin/bash

# find the best learning rate
#for DATA in mnist cifar-10; do
  #for LR in 0.0005 0.001 0.005 0.01; do
    ### find learning rate
    #sbatch hpc/exec.sh ${DATA} resnet elu --lrate ${LR} --repeat 3 --epochs 5
  #done
#done

LR=0.05
sbatch hpc/exec.sh mnist resnet elu   --lrate ${LR}
sbatch hpc/exec.sh mnist resnet relu  --lrate ${LR}

sbatch hpc/exec.sh mnist resnet nelu --lrate ${LR} --init_v 0.01
sbatch hpc/exec.sh mnist resnet nelu --lrate ${LR} --init_v 0.003
sbatch hpc/exec.sh mnist resnet nelu --lrate ${LR} --init_v 0.001
sbatch hpc/exec.sh mnist resnet nelu --lrate ${LR} --init_v 0.0003
sbatch hpc/exec.sh mnist resnet nelu --lrate ${LR} --init_v 0.0001

sbatch hpc/exec.sh mnist resnet qelu --lrate ${LR} --const --init_v 0.1
sbatch hpc/exec.sh mnist resnet qelu --lrate ${LR} --const --init_v 1.0
sbatch hpc/exec.sh mnist resnet qelu --lrate ${LR} --init_v 1
sbatch hpc/exec.sh mnist resnet qelu --lrate ${LR} --init_v 10

LR=0.05
sbatch hpc/exec.sh cifar-10 resnet elu  --lrate ${LR}
sbatch hpc/exec.sh cifar-10 resnet relu --lrate ${LR}

sbatch hpc/exec.sh cifar-10 resnet nelu --lrate ${LR} --init_v 0.01
sbatch hpc/exec.sh cifar-10 resnet nelu --lrate ${LR} --init_v 0.003
sbatch hpc/exec.sh cifar-10 resnet nelu --lrate ${LR} --init_v 0.001
sbatch hpc/exec.sh cifar-10 resnet nelu --lrate ${LR} --init_v 0.0003
sbatch hpc/exec.sh cifar-10 resnet nelu --lrate ${LR} --init_v 0.0001

sbatch hpc/exec.sh cifar-10 resnet qelu --lrate ${LR} --const --init_v 0.1
sbatch hpc/exec.sh cifar-10 resnet qelu --lrate ${LR} --const --init_v 1.0
sbatch hpc/exec.sh cifar-10 resnet qelu --lrate ${LR} --init_v 1
sbatch hpc/exec.sh cifar-10 resnet qelu --lrate ${LR} --init_v 10
