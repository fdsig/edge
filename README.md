A respository for of examples of deploying timm models on Raspberry Pi 4

of wider interset:

> -  https://petewarden.com/2021/08/05/one-weird-trick-to-shrink-convolutional-networks-for-tinyml/
> - https://github.com/jetpacapp/pi-gemm
> - https://arxiv.org/abs/2105.03536
> - https://mythic.ai/

For setting up py:

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

log in to raspberry pi and git clone repository

dowload compressed and resized ava dataset found [here](http://www.desigley.space/ava) as crushed.zip (not whole origional sized dataset). 

scp this to raspberry pi using `scp crushed.zip  host@pi4:~/crushed.zip`

```bash
ssh host@pi4
sudo apt-get install unzip
mkdir Images
mv crushed.zip Images/crushed.zip
cd Images && unzip crushed.zip
```

