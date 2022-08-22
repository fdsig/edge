#!/bin/bash
if [ "$(whoami)" != "root" ] ; then
	echo "Please run as root"
	echo "Quitting ..."
	exit 1
fi

NVME_DRIVE=$1

NEW_EXTLINUX=false

if [ -z "$NVME_DRIVE" ]
then
	echo "SSD Storage Name is Empty!!"
	exit
fi      

if [ ! -e "$NVME_DRIVE" ]; then
    echo "$NVME_DRIVE not exists!!"
	exit
fi

# Mount the SSD as /mnt
sudo mount "$NVME_DRIVE" /mnt

# Copy over the rootfs from the EMMC flash to the SSD
sudo rsync -axHAWX --numeric-ids --info=progress2 --exclude={"/dev/","/proc/","/sys/","/tmp/","/run/","/mnt/","/media/*","/lost+found"} / /mnt
# We want to keep the SSD mounted for further operations
# So we do not unmount the SSD
sync
echo 'The rootfs have copied to SSD.'

# Change the root parameter in extlinux.conf file
echo -n "Before extlinux.conf: " 
cat /mnt/boot/extlinux/extlinux.conf | grep "root="
ROOT_DRIVE=$(df '/' | tail -1 | awk '{ printf "%s", $1 }')
echo $ROOT_DRIVE > .root_drive_path.txt
sed -i 's/\//\\\//g' .root_drive_path.txt
echo $NVME_DRIVE > .nvme_drive_path.txt
sed -i 's/\//\\\//g' .nvme_drive_path.txt

if [ $NEW_EXTLINUX == true ]; then 
	sudo sed -i 's/root='$(cat .root_drive_path.txt)'/root='$(cat .nvme_drive_path.txt)'/g' /mnt/boot/extlinux/extlinux.conf
else
	sudo sed -i 's/root='$(cat .root_drive_path.txt)'/root='$(cat .nvme_drive_path.txt)'/g' /boot/extlinux/extlinux.conf
	sudo cp /boot/extlinux/extlinux.conf /mnt/boot/extlinux/extlinux.conf
fi

rm .nvme_drive_path.txt .root_drive_path.txt
echo -n "After extlinux.conf:  " 
cat /mnt/boot/extlinux/extlinux.conf | grep "root="
echo 'extlinux.conf file updated.'

echo 'Reboot for changes to take effect.'

