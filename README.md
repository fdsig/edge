A respository for of examples of deploying models (classifictiaon, object detection and semantic segmentation) On device training Nvidia Jetson, Khadas Vim3, Raspberry Pi 4. 

All configuration is for headless linux and a general philosophy of keeping as close to production OS as possible and to keep versions the same across different devices.

Here therefore we use headless Unbuntu 18.04 LTS and Ubuntu 20.04 LTS. Nvidia hardware requires flashing from Ubuntu host machine only. 

all have been set up to be accessed over ssh as host on a local/private network. 

using:

```bash
sudo apt-get install openssh-server
sudo systemctl status ssh
sudo systemctl enable --now ssh
sudo ufw allow ssh
```

This enable accesing device remotely over ssh in terminal. 


# WandB Tracking, Versioning, Deployment and Monitoring:

[here](https://wandb.ai/tinyml-hackathon)




# For setting up pi 4s:

> - build bootable disk using ubuntu disk imager utils on 
> - Instruction for installing ubuntu flavours [here](https://ubuntu.com/download/raspberry-pi)
> - plug your pi(s) into a Network Switch somthing like [this](scp inference frida@pi4:~/)
> -  plug into router 
> - find pi's ip on your local network
>> `brew install nmap`
>> `sudo nmap  192.168.1.1/24 `
>>  ssh into your pi ip pasword is on first log in `ubuntu`
> - set up paswordless ssh access via `ssh-keygen` & `ssh-copy-id`

configure python 3.7.13 used here for compatibilty and reproducibily if trining using google colab (colab's version is 3.7.13).

[pyenv](https://github.com/pyenv/pyenv) works for this on raspberry pi


```bash
sudo apt update -y

```
then:

```bash
apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
```

then (pyenv and pyenv virtualenv):

```bash
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.bashrc
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
```

finally:

```bash
exec "$SHELL"
```

to create verison of python

```pyenv install 3.10.1
pyenv virtualenv 3.10.1 edge
```


log in to raspberry pi and git clone repository



## For Classification inference

dowload compressed and resized ava dataset found [here](http://www.desigley.space/ava) as crushed.zip (not whole origional sized dataset). 

scp this to raspberry pi using `scp crushed.zip  host@pi4:~/crushed.zip`

```bash
ssh host@pi4
sudo apt-get install unzip
mkdir Images
mv crushed.zip Images/crushed.zip
cd Images && unzip crushed.zip
```


# Khadas Vim3 (NPU)

- [x] Set up and clone, using ssd M2 

- [ ] benchmark cpu
- [ ] port models to onnx
- [x] change boot partition to ssd
- [x] update inference.py to torchvision (albumentations not needed for inference)
- [ ] remove sklearn dependency for inference ( should training be a part of edge repo at all?)
- [ ] comment code file reading section fo main.py
- [ ] downgrade to python 3.6 

#Flashing system/setting up for first use (Nvidia SDK Manager and Auvidea Carrier board)

Auvidea carried board with Jeston (nano, tx xn) can be found [here](https://www.google.com/url?q=https://auvidea.eu/download/QuickStart.pdf&sa=U&ved=2ahUKEwjVi_b21Mb5AhWJhP0HHWYxBhEQFnoECAoQAg&usg=AOvVaw3gJ2ZtS91IDzuISZQC_bm8)
 and gives instructions about how to flash device when in forced recovery mode.



# Camera Streaming

to do:
- [x] stream from camera using ffmpeg and transformt bytes to array
- [ ] get running version that can handle recording/ streaming for many hours
- [ ] embed inferece into streaming process and establish bottlencks

# Nvidia Jetson NX 8 GB






To do
- [ ] benchmark inferenc
- [ ] benchmanrk on device trining
- [ ] stream from camera to jetson
- [ ] deploy in the wild
- [ ] transfer learning on manually labeled dataset
- [ ] train weather classifier from images. 


## of wider interset:

> - https://petewarden.com/2021/08/05/one-weird-trick-to-shrink-convolutional-networks-for-tinyml/
> - https://github.com/jetpacapp/pi-gemm
> - https://arxiv.org/abs/2105.03536
> - https://mythic.ai/
> - https://developer.nvidia.com/embedded/jetson-tx2
> - https://ffmpeg.org/

