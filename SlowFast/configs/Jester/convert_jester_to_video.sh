# wdf write to convert jester image dataset to video
for ((i=1;i<=148092;i++))
do
        img_folder="/data/20bn-jester-v1/$i"

        # whether the folder exists?
        if [ ! -d $img_folder ]; then
                echo $img_folder >> /data/not_exist_folders.error
        fi

        echo $i, $img_folder, $img_folder/$i.mp4
        ffmpeg -start_number 1 -r 12 -i $img_folder/%05d.jpg -vcodec mpeg4 $img_folder/$i.mp4
done
