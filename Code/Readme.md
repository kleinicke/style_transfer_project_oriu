# Styletransfer

To run the style transfer for images run the file fotostyle.sh. If you miss some required arguments it will tell you. Run it f.e. with the following command.
./transferStyle.sh images/content1.jpg style2

The style transfer itself happens in stylize.py.
If you want to run it directly you can so. Orientate yourself on the usage in transferStyle.sh. You can run it with the comment

python stylize.py --content test6 --style style3 --size 384 512

To perform a style transfer on a video run transferStyle.sh.
It will also tell you when you missed an argument. Run it like the following.

./transferStyle.sh videos/vid2.mp4 images/style1.jpg no

The no stands for whether the init frames should be warped or not.
Just don't use the third argument, then the init frames will be warped.

To run all the simulations with the different configurations that were mentioned in the report, the script has to be changed. Just comment out specific arguments in line 108 or 110 to run the style transfer the way you want to.
To change the number of iteration change them in line 31 and 32.
The resolution is asked everytime you run the script.

For more information about what happens in each file, please read the report I handed in.