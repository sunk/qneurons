#!/bin/bash

# find the best learning rate
#for DATA in mnist cifar-10; do
  #for LR in 0.01 0.05 0.1 0.5 1.0; do
    ### find learning rate
    #sbatch hpc/exec.sh ${DATA} ae tanh --lrate ${LR} --repeat 3 --epochs 5
  #done
#done

LR=0.1
sbatch hpc/exec.sh mnist ae tanh  --lrate ${LR}
sbatch hpc/exec.sh mnist ae tanh  --lrate ${LR} --dropout 1

sbatch hpc/exec.sh mnist ae ntanh --lrate ${LR} --init_v 0.0001
sbatch hpc/exec.sh mnist ae ntanh --lrate ${LR} --init_v 0.0003
sbatch hpc/exec.sh mnist ae ntanh --lrate ${LR} --init_v 0.001
sbatch hpc/exec.sh mnist ae ntanh --lrate ${LR} --init_v 0.003

sbatch hpc/exec.sh mnist ae qtanh --lrate ${LR} --const --init_v 0.1
sbatch hpc/exec.sh mnist ae qtanh --lrate ${LR} --const --init_v 1.0
sbatch hpc/exec.sh mnist ae qtanh --lrate ${LR} --init_v 1
sbatch hpc/exec.sh mnist ae qtanh --lrate ${LR} --init_v 10

LR=0.5
sbatch hpc/exec.sh cifar-10 ae tanh  --lrate ${LR}
sbatch hpc/exec.sh cifar-10 ae tanh  --lrate ${LR} --dropout 1

sbatch hpc/exec.sh cifar-10 ae ntanh --lrate ${LR} --init_v 0.0001
sbatch hpc/exec.sh cifar-10 ae ntanh --lrate ${LR} --init_v 0.0003
sbatch hpc/exec.sh cifar-10 ae ntanh --lrate ${LR} --init_v 0.001
sbatch hpc/exec.sh cifar-10 ae ntanh --lrate ${LR} --init_v 0.003

sbatch hpc/exec.sh cifar-10 ae qtanh --lrate ${LR} --const --init_v 0.1
sbatch hpc/exec.sh cifar-10 ae qtanh --lrate ${LR} --const --init_v 1.0
sbatch hpc/exec.sh cifar-10 ae qtanh --lrate ${LR} --init_v 1
sbatch hpc/exec.sh cifar-10 ae qtanh --lrate ${LR} --init_v 10
