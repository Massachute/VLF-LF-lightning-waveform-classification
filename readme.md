## **Attention!**  
These files were uploaded **via git LFS**. Do not use "code"-"download zip" to download files.  
You can try these methods:  
1.download one by one  
- click the filename of the file you want to download and enter the file tree interface.  
- Download raw file by click the download buttom besides the "raw" buttom. Or use "ctrl+shift+s" to start downloading.
![image](https://github.com/Massachute/VLF-LF-lightning-waveform-classification/assets/47164880/17b028e5-8c17-4344-b67a-3d685412fa2c)

2.Use git in your command shell 
- cd to your directory  
- git clone https://github.com/Massachute/VLF-LF-lightning-waveform-classification.git  
- git lfs pull

## **Description**
This repo includes 8000 lightning waveforms stored in .pkl format.  
There are four types including RS, PB, NB and IC with 2000 waveforms for each.  
Use pickel in python to open the pkl file. A numpy array of 8000 * 10001 is in this file.  
Each row has 10000 datapoints and 1 label type number (0-RS,1-PB,2-NB,3-IC)  
The picture for each waveform is plotted and can be seen in Plots.zip.  


  
