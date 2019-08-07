Practical - Installation of software packages and libraries (R and Python keras) 
=================================================================================

Download the examples
---------------------
1. Clone the https://github.com/nknewlands/DeepLearningShortCourse into your laptop.



Keras + Tensorflow GPU + python* 
--------------------------------

1. Installation on Linux
   Install miniconda for python 3.7 (https://docs.conda.io/en/latest/miniconda.html) following the default options for your OS.
2. Open a terminal
3. Type: `conda create -y --name deeplearning` to create a new conda environment
4. Type: `conda activate deeplearning` to activate this new environment
5. Type: `conda install -c anaconda -y keras pandas scikit-learn spyder opencv`
 OR 
          `conda install -c anaconda -y keras-gpu pandas scikit-learn spyder opencv`
6. Type: `conda install -y -c conda-forge matplotlib`

*Alternative: Use google collaboratory https://colab.research.google.com/

Keras + Tensorflow + R using RStudio
------------------------------------

1. Go to https://cran.r-project.org/ and download R found in the “base” section of the “Windows” or other distribution.
2. Follow the installer instruction using the default options
3. Download the free RStudio desktop from https://www.rstudio.com/products/rstudio/download/
4. Follow the installation instructions using the default options
5. Once installed, start RStudio
6. Go into the *console* section of the RStudio IDE
6. Type: `install.packages("keras")`
7. Type: `library(keras)`
8. Type: `install_keras()` or `install_keras(tensorflow = "gpu")`
9. Type: `install.package(“imager”)`

Test the deep learning environment using the MNIST dataset [LeCun et al. 1998]
-------------------------------------------------------------------------------
* Note: An internet connection is require to download this dataset *

The MNIST dataset is a set of 28 x 28 pixels images of handwritten digits (0-9) classified into 10 classes. 
This is considered as the * Hello world * of deep learning.and contains 70,000 annotated images. 
A normal accuracy of >98% is expected using the default parameters.

![Sample image of the MNIST dataset][MNIST]

Keras + Tensorflow GPU + python
--------------------------------
1. Go into the Day1\Installations
2. Open a command line prompt
3. Run: `python test_keras.py`

Keras + Tensorflow + R using RStudio
------------------------------------

1. Open RStudio and navigate to the Session -> Set working directory -> Choose Directory
![Changing the RStudio session][RStudio]
2. Select the directory Day1\Installations
3. Open the file `test_keras.R`


References----------
[LeCun et al. 1998] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998. (http://yann.lecun.com/exdb/mnist/)
[RStudio]: https://github.com/nknewlands/DeepLearningShortCourse/raw/master/Day1/Installations/RStudio_session.png "Select session in R Studio"
[MNIST]:  https://github.com/nknewlands/DeepLearningShortCourse/raw/master/Day1/Installations/mnist.png "Mnist"
