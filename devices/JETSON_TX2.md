## Istructions for setting up hardware and Sofwtwar on nvidia Jetson

Nvidia Software developer kit (SDK) is called Jetpack and can be found [here](https://developer.nvidia.com/embedded/jetpack). 

The Nvidia jetson Nano looks like this, well inside an aluminum protection and cooling case:

from the side:

![](jetson_tx_2.png.png)

From the top:

![](tx_2_fan.jpg.jpg)

This is unmounted and on the bottom you can see the (peripheral component interconnect express)pcie slots 

The Auvdea is a german mad base board, the Jetson has jetpack (linux os) pre installed and has a different setup process to other single board computers. Base board is support. 

The setup unlike other single board does not require flashing an ssd or 2m 
physical setup, 

attache the jetson nano via pcie lanes and screw together, connect fan JST (Japan Solderless Terminal) micro 1.25mm pitch connector (4 pin)
connect micro hdmi,
connect usb bluetooth mouse, keyboard adaptor. 

The system automatically boots, debugging any issues with this need to happen with a ttl converter Universal Asynchronos Receiver/Transmitter UART output. 


full technical details can be found [here](https://auvidea.eu/download/manual/J120/J120_J121_technical_reference_2.0.pdf)

Software setup:

This is far simpler and requires simply booting, NVIDIA's own os loads, My own preference is for headless setups however however given pre installed UI is present and flashing the drive is more complex than other single board setups it is easier to proceed with the system as is and then ssh into a headless boot. 


