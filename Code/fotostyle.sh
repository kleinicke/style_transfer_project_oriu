set -e
# Get a carriage return into `cr`
cr=`echo $'\n.'`
cr=${cr%.}

#File to stylize a single image.
#if you run it, it will tell you the arguments (with working example) it needs.


if [ "$#" -le 1 ]; then
   echo "Usage: ./fotostyle.sh <path_to_video> <path_to_style_image>"
   echo "./fotostyle.sh images/content1.jpg style2"
   exit 1
fi

# Parse arguments
filename=$(basename "$1")
extension="${filename##*.}"
filename="${filename%.*}"
filename=${filename//[%]/x}
style_image=$2


#set number of steps
steps=1000

#set resolution
resolution=525:350
#resolution=600:400

sizew=$(echo $resolution | cut -f1 -d:)
sizeh=$(echo $resolution | cut -f2 -d:)
echo $sizew
echo $sizeh



# Save frames of the video as individual image files
startTime="$(date +%s)"



for stylename in images/style[1-6].jpg
do
    stylename=$(basename "$stylename")
    stylename="${stylename%.*}"
    stylename=${stylename//[%]/x}
    echo python stylize.py --content ${filename} --style ${stylename} --size $sizeh $sizew --steps ${steps} --savesteps
    python stylize.py --content ${filename} --style ${stylename} --size $sizeh $sizew --steps ${steps} --savesteps
done


echo " "
echo " "
echo "Time in seconds for everything: $(($(date +%s)-startTime))"
echo " "
echo " "

exit 1
