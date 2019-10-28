set -e
# Get a carriage return into `cr`
cr=`echo $'\n.'`
cr=${cr%.}





if [ "$#" -le 1 ]; then
   echo "Usage: ./stylizeVideo <path_to_video> <path_to_style_image><optout_flow"
   echo "./transferStyle.sh videos/vid2.mp4 images/style1.jpg no"
   exit 1
fi

# Parse arguments
filename=$(basename "$1")
extension="${filename##*.}"
filename="${filename%.*}"
filename=${filename//[%]/x}
style_image=$2

stylename=$(basename "$2")
stylename="${stylename%.*}"
stylename=${stylename//[%]/x}
echo $stylename
style=$stylename #style3

flow=$(basename "$3")

steps1=600
steps2=300

echo $flow
if [ -z "$flow" ]; then
  echo "With flow"
else
  echo "Without flow"
fi

echo ""
  read -p "Enter resolution of video h:w\
  example 128:128, 260:150, 175:100 or 525:350$cr > " resolution



sizew=$(echo $resolution | cut -f1 -d:)
sizeh=$(echo $resolution | cut -f2 -d:)
echo $sizew
echo $sizeh

#ffmpeg -i output/video/${filename}/flowimage_%04d.png output/video/style_${filename}_${sizeh}_${sizew}.mp4
#ffmpeg -i output/video/${filename}/stylized-${filename}-${style}-${sizeh}_${sizew}-v%04d.png output/video/${filename}_${sizeh}_${sizew}.mp4

#ffmpeg -i videos/${filename}flow/initImage_%04d.png output/video/${filename}_stylenflow_${sizeh}_${sizew}.mp4

#exit 1
#Create output folders and deleates previous generated files
mkdir -p videos/$filename
rm -rf videos/$filename/*
mkdir -p videos/${filename}flow
#rm -rf videos/${filename}flow/*
mkdir -p output/video/$filename


# Save frames of the video as individual image files
#TIMEFORMAT='It takes %R seconds to decode video into images'
#time {
startTime="$(date +%s)"


#creates images out of video
if [ -z $resolution ]; then
  ffmpeg -i $1 videos/${filename}/frame_%04d.ppm
  resolution=default
else
  ffmpeg -i $1 -vf scale=$resolution videos/${filename}/frame_%04d.ppm
fi
echo $resolution

echo " "
echo " "
echo "Time in seconds for decoding: $(($(date +%s)-startTime))"
echo " "
echo " "


##create flow + flow video
python create_flow.py --name ${filename}
ffmpeg -i output/video/${filename}/flowimage_%04d.png -y output/video/style_${filename}_${sizeh}_${sizew}.mp4


python stylize.py --content ${filename} --style ${style} --size $sizeh $sizew --video --steps ${steps1} --frame 0001

#for i in `seq 2 10`;
for i in videos/${filename}/frame_*.ppm
do
  string="${i//[!0-9]/}"
  j=${string: -4}
  one='0001'
  echo $j
  if [ "$j" == "$one" ]
    then
    continue
  fi
  if [ -z "$flow" ]; then
    python apply_flow.py --name ${filename} --style ${style} --imsize $sizeh $sizew --toPos $j
    python stylize.py --content ${filename} --style ${style} --size $sizeh $sizew --video --steps ${steps2} --frame $j  --init --memory --flow
  else
    python stylize.py --content ${filename} --style ${style} --size $sizeh $sizew --video --steps ${steps2} --frame $j --init
  fi

done


if [ -z "$flow" ]; then
  ffmpeg -i output/video/${filename}/stylized-${filename}-${style}-${sizeh}_${sizew}-v%04d.png output/video/${filename}_${style}_${sizeh}_${sizew}_with_flowinit_$(($(date +%s)-startTime)).mp4
  ffmpeg -i videos/${filename}flow/content_flow_%04d.png output/video/${filename}_contentflow_${style}_${sizeh}_${sizew}_with_flowinit_$(($(date +%s)-startTime)).mp4
else
  ffmpeg -i output/video/${filename}/stylized-${filename}-${style}-${sizeh}_${sizew}-v%04d.png output/video/${filename}_${style}_${sizeh}_${sizew}_no_flow_$(($(date +%s)-startTime)).mp4
fi
#ffmpeg -i output/video/${filename}/stylized-${filename}-${style}-${sizeh}_${sizew}-v%04d.png output/video/${filename}_stylenflow_${sizeh}_${sizew}.mp4
#ffmpeg -i videos/vid2_4flow/content_flow_%04d.png output/video/$vid2_4_contentflow.mp4

echo " "
echo " "
echo "Time in seconds for everything: $(($(date +%s)-startTime))"
echo " "
echo " "

exit 1
