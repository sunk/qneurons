#!/bin/bash

# find the best learning rate
#for DATA in mnist cifar-10; do
  #for LR in 0.001 0.005 0.01 0.05 0.1 0.5; do
    ## find learning rate
    #sbatch hpc/exec.sh ${DATA} siamese elu --lrate ${LR} --repeat 3 --epochs 5
  #done
#done

LR=0.05
sbatch hpc/exec.sh mnist siamese elu   --lrate ${LR}
sbatch hpc/exec.sh mnist siamese elu   --lrate ${LR} --dropout 1
sbatch hpc/exec.sh mnist siamese relu  --lrate ${LR}

sbatch hpc/exec.sh mnist siamese nelu --lrate ${LR} --init_v 0.01
sbatch hpc/exec.sh mnist siamese nelu --lrate ${LR} --init_v 0.003
sbatch hpc/exec.sh mnist siamese nelu --lrate ${LR} --init_v 0.001
sbatch hpc/exec.sh mnist siamese nelu --lrate ${LR} --init_v 0.0003
sbatch hpc/exec.sh mnist siamese nelu --lrate ${LR} --init_v 0.0001

sbatch hpc/exec.sh mnist siamese qelu --lrate ${LR} --const --init_v 0.1
sbatch hpc/exec.sh mnist siamese qelu --lrate ${LR} --const --init_v 1.0
sbatch hpc/exec.sh mnist siamese qelu --lrate ${LR} --init_v 1
sbatch hpc/exec.sh mnist siamese qelu --lrate ${LR} --init_v 10

LR=0.05
sbatch hpc/exec.sh cifar-10 siamese elu  --lrate ${LR}
sbatch hpc/exec.sh cifar-10 siamese elu  --lrate ${LR} --dropout 1
sbatch hpc/exec.sh cifar-10 siamese relu --lrate ${LR}

sbatch hpc/exec.sh cifar-10 siamese nelu --lrate ${LR} --init_v 0.01
sbatch hpc/exec.sh cifar-10 siamese nelu --lrate ${LR} --init_v 0.003
sbatch hpc/exec.sh cifar-10 siamese nelu --lrate ${LR} --init_v 0.001
sbatch hpc/exec.sh cifar-10 siamese nelu --lrate ${LR} --init_v 0.0003
sbatch hpc/exec.sh cifar-10 siamese nelu --lrate ${LR} --init_v 0.0001

sbatch hpc/exec.sh cifar-10 siamese qelu --lrate ${LR} --const --init_v 0.1
sbatch hpc/exec.sh cifar-10 siamese qelu --lrate ${LR} --const --init_v 1.0
sbatch hpc/exec.sh cifar-10 siamese qelu --lrate ${LR} --init_v 1
sbatch hpc/exec.sh cifar-10 siamese qelu --lrate ${LR} --init_v 10
