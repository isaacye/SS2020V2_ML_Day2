# SS2020V2_ML_Day2
--------------------------------------------------------------------------------

Day 2 will provide an overview of Artificial Intelligence with a focus on Deep Learning (DL) and Deep Neural Networks (DNN). We will have hands-on tutorials on several popular DL problems such as Linear regression and classification cases. Mathematical background for supervised learning in DL will be discussed and fundamental foundation for advanced techniques will be built through step-by-step approaches. Hands-on exercises with PyTorch will be practiced on Google Colab platform and will be extended to run a DL code on GPU nodes in Graham system of Compute Canada.

**Contents**
* [Session 1](https://github.com/isaacye/SS2020V2_ML_Day2#Session-1) : Introduction to Deep learning/ Single variable linear regression problem
* [Session 2](https://github.com/isaacye/SS2020V2_ML_Day2#Session-2) : Multi-variable linear regression problem
* [Session 3](https://github.com/isaacye/SS2020V2_ML_Day2#Session-3) : Multi-Layer Perceptron
* [Session 4](https://github.com/isaacye/SS2020V2_ML_Day2#Session-4) : Convolutional Neural Network

--------------------------------------------------------------------------------
## Session 1 (10:00 AM - 11:00 AM) <!-- : [[Lecture slide]](https://github.com/isaacye/SS2020_ML_Day2/blob/master/Session_2/SS20_ML_Day2_Session%20I.pdf) -->

**What we cover**
* Short introduction to Deep Learning (DL) and framework.
* Single variable linear regression problem
* Cost function/ gradient decent algorithm / learning rate

### :computer: Lab 1:  Single variable linear regression (vanilla) [[Demo code]](https://github.com/isaacye/SS2020V2_ML_Day2/blob/master/Session_1/SS20_lab1_LR_vanilla.ipynb)

--------------------------------------------------------------------------------
## Session 2 (11:00 AM - 12:00 PM) <!--: [[Lecture slide]](https://github.com/isaacye/SS2020_ML_Day2/blob/master/Session_2/SS20_ML_Day2_Session%20II.pdf) -->

**What we cover**
* PyTorch modules (linear model, cost function- MSE, optimizer - SGD)
* Running a DL code on Graham in Compute Canada



### :computer: Lab 2A: Multivariable linear regression (PyTorch) [[Demo code]](https://github.com/isaacye/SS2020V2_ML_Day2/blob/master/Session_2/SS20_lab2_LRm.ipynb)

###  :computer: Lab 2B: Running DL codes in Graham ###

#### Working environment in Graham ####
1. Log into graham.computecanada.ca with guest account and p/w : please see [[this page]](https://docs.computecanada.ca/wiki/SSH) for further details.

   (Use MobaXterm or Putty for Windows / Open terminal in Linux or Mac)

2. Load modules and make a virtual environment: please see [[this page]](https://docs.computecanada.ca/wiki/Python#Creating_and_using_a_virtual_environment) for further details.

   ```
   module load python
   module load scipy-stack
   virtualenv --no-download ~/ENV
   ```
3. Activate virtual enviornment and upgrade/install Pip and PyTorch: please see [[this page]](https://docs.computecanada.ca/wiki/PyTorch#Installation) for further details.
   ```
   source ~/ENV/bin/activate
   pip install --no-index --upgrade pip
   pip install --no-index torch
   pip install --no-index torchvision torchtext torchaudio
   pip install sklearn
   ```
4. (Optional) Deactivate virtual enviornment
   ```
   deactivate
   ```

#### Running a simple DL code in Graham ####
1. Download **`Lab2A_Linear_Reg_Vanilla.ipynb`** as .py file from Google Colab

2. File transfer **`Lab2A_Linear_Reg_Vanilla.py`** to Graham using WinScp or MobaXterm (Windows) / sftp (Linux, Mac): please see [[this page]](https://docs.computecanada.ca/wiki/Transferring_data#SFTP) for further details.

3. Activate virtual environment (make sure you load python and scipy-stack module)
   ```
   source ~/ENV/bin/activate
   ```
4. Run it
   ```
   python Lab2A_Linear_Reg_Vanilla.py
   ```
5. Note you need to collect all import commands into the beginning of code using text editor (Nano/emacs/VI): Please see [[Nano basic]](https://wiki.gentoo.org/wiki/Nano/Basics_Guide) for further details.

6. Note that you need to save/close your plots with proper filename for each plotting command. (You may see what I did [[here.]](https://github.com/isaacye/SS2020_ML_Day2/blob/master/Session_3/Lab3A_linear_Reg_Vanilla_Graham.py))

7. File transfer plotting files to your local computer using WinScp or MobaXterm (Windows) / sftp (Linux, Mac) and check it out


--------------------------------------------------------------------------------
## Session 3 (1:30 PM - 15:30 PM) <!--: [[Lecture slide]](https://github.com/isaacye/SS2020_ML_Day2/blob/master/Session_3/SS20_ML_Day2_Session%20III.pdf) -->

**What we cover**
* Binary classification
* Logistic model, cross entropy function
* Issue with linear regression
* Activation function
* XOR problem
* Multi-layer Perceptron
* GPU on Graham



### :computer: Lab 3A:  Linear regression (MLP) [[Demo code]](https://github.com/isaacye/SS2020_ML_Day2/blob/master/Session_3/Lab3B_Linear_Reg_MLP.ipynb)

#### Running a DL code on `CPU` _interactively_ in Graham ####

1. Download **`Lab3B_Linear_Reg_MLP.ipynb`** as .py file from Colab

2. File transfer **`Lab3B_Linear_Reg_MLP.py`** to Graham using WinScp or MobaXterm (Windows) / sftp (Linux, Mac)

3. Start interactive running mode 
   ```
    salloc --time=0:30:0 --ntasks=1 --cpus-per-task=3 --nodes=1 --mem=1000M --account=def-training-wa --reservation=snss20_wr_cpu
   ```

4. virtual environment (make sure you load python and scipy-stack module)

    ```
    module load python
    module load sci-py-stack
    source ~/ENV/bin/activate
    ```

5. Run it 
    ```
    python Lab3B_Linear_Reg_MLP.py
    ```
    
6. Note you need to collect all import commands into the beginning of code using text editor (Nano/emacs/VI)

7. Note that you need to save/close your plots with proper filename for each plotting command like below

8. File transfer plotting files to your local computer using WinScp or MobaXterm (Windows) / sftp (Linux, Mac) and check it out

### :computer: Lab 3B:  Linear regression (MLP) with GPU [[Demo code]](https://github.com/isaacye/SS2020_ML_Day2/blob/master/Session_3/Lab3C_Linear_Reg_MLP_GPU.ipynb)

#### Running a DL code on `GPU` _interactively_ in Graham ####

1. Copy Lab3C_Linear_Reg_MLP_GPU.py from /home/isaac/SS20_ML_Day2
    ```
    cp /home/isaac/SS20_ML_Day2/Lab3C_linear_Reg_MLP_GPU.py /home/$USER
    ```

2. Start interactive running mode with T4 GPU in Graham 
   ```
    salloc --time=0:30:0 --ntasks=1 --cpus-per-task=3 --reservation=snss20_wr_gpu --gres=gpu:t4:1 --nodes=1 --mem=1000M --account=def-training-wa_gpu
   ```

3. virtual environment (make sure you load python and scipy-stack module)

    ```
    module load python
    module load sci-py-stack
    source ~/ENV/bin/activate
    ```

4. Run it 
    ```
    python Lab3C_linear_Reg_MLP_GPU.py
    ```
    
5. File transfer plotting files to your local computer using WinScp or MobaXterm (Windows) / sftp (Linux, Mac) and check it out


#### Running a DL code _via scheduler_ in Graham ####

1.  Write a submission script 'job_s.sh' like below using text editor  
    ```
    #!/bin/bash
    #
    #SBATCH --nodes=1
    #SBATCH --gres=gpu:t4:1
    #SBATCH --mem=20000M
    #SBATCH --time=0-30:00
    #SBATCH --account=def-training-wa_gpu
    #SBATCH --output=slurm.%x.%j.out
    
    module load python scipy-stack
    source ~/ENV/bin/activate
    python Lab3C_Linear_Reg_MLP_GPU.py
    
    ```
    
4. Submit it
    ```
    sbatch job_s.sh
    ```

5. Check the submitted job
    ```
    squeue -u $USER
    ```
    
6. Note you may need to collect all import commands into the beginning of code using text editor (Nano/emacs/VI)

7. Note that you may need to save/close your plots with proper filename for each plotting command like below

8. File transfer plotting files to your local computer using WinScp or MobaXterm (Windows) / sftp (Linux, Mac) and check it out


--------------------------------------------------------------------------------
## Session 4 (15:30 PM - 17:00 PM) <!--: [[Lecture slide]](https://github.com/isaacye/SS2020_ML_Day2/blob/master/Session_4/SS20_ML_Day2_Session%20IV.pdf) -->

**What we cover**
* Backward propagation, model capacity
* Overfitting, vanishing gradient problem
* MNIST image classification
* Vanishing gradient problem
* Issue with MLP
* Convolutional neural network (CNN)

### :computer: Lab4A: MNIST Image classification [[Demo code]](https://github.com/isaacye/SS2020_ML_Day2/blob/master/Session_4/Lab4A_MNIST_classification.ipynb)

#### :computer: Lab4B: CIFAR-10 Classification (pytorch) [[Demo code]](https://github.com/isaacye/SS2020_ML_Day2/blob/master/Session_4/Lab4B_CIFAR10_classification.ipynb)

## Acknowledgement

KAIST Idea Factory - [[Deep learning alone]](https://github.com/heartcored98/Standalone-DeepLearning)

Stanford [[CS231n]](http://cs231n.stanford.edu/)

Pytorch document [[1.5.0]](https://pytorch.org/docs/stable/index.html)
