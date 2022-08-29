Discussed elsewhere there are a number of differnet ways set up jetson module on cerried boreads.
here we are using a an Auvidea Jnx30 
carrier board which is compatible with (Nano, TX nx, and Xavier nx)

All of these board have the same for factor and Slot into carrier board (insert slot type)


get firmware from Auvidea

Auvidea  firmware can be found [here](https://auvidea.eu/firmware/)

the command for Jetson firmware is: 
```bash
wget https://f000.backblazeb2.com/file/auvidea-download/images/Jetpack_4_6/BSP/Jetpack4.6_Nano_BSP.tar.gz
```

Auvidea quick-start guide can be found [here](https://auvidea.eu/download/QuickStart.pdf)


Installing pytorch instructions for Jetson can be found [here](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html#prereqs-install)

```bash
sudo apt-get -y update; 
sudo apt-get -y install autoconf bc build-essential g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl python-openssl rsync scons python3-pip libopenblas-dev;

```

some additional installs are required before install pytorch in python3.

```
sudo apt-get install libopenblas-base libopenmpi-dev 
```


the Nvidia developer forums are an invaluable sources of troubleshooting for example [here](https://forums.developer.nvidia.com/t/cannot-install-pytorch/149226/5?u=fdesigley) are additional commands required for running Pytroch on Jetson nano.

moving to boot partitions using .sh script to ssd rather than device mmc.



If all of the above has been done correctly

simply running python3 and import Pytorch should yield true:

```
import torch
torch.cuda.is_available()
```
To do:

- [ ] Back compatible Inference scrip to python 3.6.9
- [ ] Torch vision install main python from wheel on Jetson
- [ ] ta to device 



