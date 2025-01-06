#!/bin/bash -l

set -eu

today=$(date +%Y-%m-%d)
output_dir=/bird-detections/$today
vid_duration=15

. ~/rknn-venv/bin/activate

# 1. find best videos
bestvids=$(~/birds/scripts/find-best-vids.py --csv "$output_dir/meta.csv" --num-videos 5 --duration $vid_duration)

# 2. create gifs from best videos
gifs2stack=""
for file2gif in $bestvids
do
	f2g=${file2gif%.*}.mp4
	fname="$(basename ${f2g})"
	fname="${fname%.*}"
	duration=$(ffmpeg -i "$f2g" 2>&1 | grep "Duration" | cut -d ' ' -f 4 | sed s/,// | sed 's@\\..*@@g' | awk '{ split($1, A, ":"); print 3600*A[1] + 60*A[2] + B[1] }')
	let "starttime = $duration/2 - $vid_duration"
	[ "$starttime" -lt 0 ] && starttime=0
	ffmpeg -y -ss "$starttime" -t $vid_duration -i "$f2g" \
		-vf "scale=480:-1:flags=lanczos" \
		/tmp/"$fname.gif"
	gifs2stack="/tmp/$fname.gif $gifs2stack"
done

# 3. stack gifs
~/birds/scripts/stack-gifs.py /tmp/stacked.gif $gifs2stack

# 4. optimize gif
gifsicle -O3 --lossy=35 -i /tmp/stacked.gif --colors 128 -o "$output_dir/sample.gif"

# 5. send email
## create google drive link to triggers folder
## save list of birds
birblist="$(~/birds/scripts/parse-instances.py "$output_dir/meta.csv")"
## send email
export GMAIL_APP_PWD=bpcshpyjugjpmbvy
~/birds/scripts/send_birb_summary.py \
	-s "Birb watcher update - $today" \
	-i "$output_dir/sample.gif" \
	eds.birb.watcher@gmail.com \
	"Birb Watcher" \
	birds/email-lists.csv \
	"<html>
<body>
	<p>Hi,</p>
	<p>I've been recording videos all day, and across all video frames I saw:</p>
	<ul>
	$birblist
	</ul>
<p>Here's one of the videos with birds:</p>
	<img src=\"cid:{image_cid}\">
	<p>Hope you have a great day!</p>
	<p>Regards,<br>Ed's Birb Watcher</p>
</body>
</html>"

# 6. upload to google drive
#rclone -P move ~/bird-detections gdrive:birds/$today

