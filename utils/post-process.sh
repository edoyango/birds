#!/bin/bash -l

set -eu

today=$(date +%Y-%m-%d)
output_dir=/output/$today
vid_duration=15

# . ~/rknn-venv/bin/activate

# 1. find best videos
bestvids=$(/app/find-best-vids.py --csv "$output_dir/meta.csv" --num-videos 5 --duration $vid_duration)

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
/app/stack-gifs.py /tmp/stacked.gif $gifs2stack

# 4. optimize gif
gifsicle -O3 --lossy=35 -i /tmp/stacked.gif --colors 128 -o "$output_dir/sample.gif"

# 5. render panel
curl -o "$output_dir/panel.png" 'admin:orangepi@grafana:3000/render/d-solo/de8twne7u2r5sc?orgId=1&from=now-12h&to=now&panelId=10&width=720&height=480&tz=Australia%2FSydney'

# 5. send email
## create google drive link to triggers folder
## save list of birds
birblist="$(/app/parse-instances.py "$output_dir/meta.csv")"
## send email
export GMAIL_APP_PWD=bpcshpyjugjpmbvy
/app/send_birb_summary.py \
	-s "Birb watcher update - $today" \
	eds.birb.watcher@gmail.com \
	"Birb Watcher" \
	/app/email-lists.csv \
	"<html>
<body>
	<p>Hi,</p>
	<p>I've been recording videos all day, and across all video frames I saw:</p>
	<ul>
	$birblist
	</ul>
<p>Here's a heatmap of all the birds I saw throughout the day!</p>
	<img src=\"cid:{image_cid1}\">
<p>And here's one of the videos with birds:</p>
	<img src=\"cid:{image_cid0}\">
	<p>Hope you have a great day!</p>
	<p>Regards,<br>Ed's Birb Watcher</p>
</body>
</html>" \
	-i "$output_dir/sample.gif" "$output_dir/panel.png"

# 6. upload to google drive
#rclone -P move ~/bird-detections gdrive:birds/$today

