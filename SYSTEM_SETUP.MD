# This section covers setting up edge devices where setting up aspects such as mounting emmc m2  ssd so that data can be stored.

```bash 
sudo lsblk
```

wish show currnet mounted drives. The easiest way to find your ssd inserted is to run this before and after inserting the ssd.

perhaps storing on or both of the lsblk outputs using `sudo lsblk > drives.txt` .

on. here the example is 'nvme0n1'
```
mkdir ssd
mount /dev/nvme0n1 ssd/
```
Will mount this.

You might also want to format your drive using :

```bash 
sudo mkfs.ext4 /dev/nvme0n1
```
and then mount. Formatting your SSD will also give you a UUID that you can use in adding to fstab file which will mount this drive on boot. 


On an edge device that is headless (no UI) you can probably call this somthing like data or name it after where you actuall want to store data from live inferencing recorded data.

in order to make this persist on each boot you will need to edti the /etc/fstab

with `vim /etc/fstab` . adding your UUID.

If not formatting driev then you can get UUID using:

`ls -l /dev/disk/by-uuid` or `lsblk -f` or `sudo blkid /dev/nvme0n1` more on this can be found [here](https://liquidat.wordpress.com/2013/03/13/uuids-and-linux-everything-you-ever-need-to-know/)


```bash
# <file system>                           <mount point>   <type>          <options>                <dump> <pass>
/dev/root                                 /                ext4           defaults                      0 1
UUID=xxxx-xxxx-xxxx-xxxx /home/user_nam/images  ext4           defaults                      0 2 
```