#!/bin/bash

set -ue

script_dir=$(dirname "$0")

email_list=$1

# save the date in yyyy-mm-dd format into the "date" variable
printf -v date '%(%Y-%m-%d)T' -1

# create a sample
python3 "$script_dir"/stack_sample_images.py observations/$date/triggers observations/$date/sample.jpg

# copy new files
rclone copy observations/ google-drive-birds:/

exit 0

# create a public, read-only link to the folder
drive_link=$(rclone link google-drive-birds:/$date)

nbirbs=$(find observations/$date/instances -type f -name 'im.jpg*.jpg' | wc -l)

birblist=$(for species in observations/"$date"/instances/*/; do echo "<li>$(basename "$species"): $(find "$species" -type f | wc -l)</li>"; done)

python3 "$script_dir"/send_birb_summary.py \
    -s "Birb watcher update" \
    -i observations/"$date"/sample.jpg \
    eds.birb.watcher@gmail.com \
    "Birb Watcher" \
    "$email_list" \
    "<html>
    <body>
        <p>Hi {FIRST},</p>
	<p>I saw ${nbirbs} today! These were:</p>
        <ul>
        $birblist
        </ul>
        <p>You can view all the birds I saw today via <a href=\"${drive_link}\">this google drive link</a>!</p>
        <p>The \"triggers\" folder shows what the AI model thought was a bird.<br>The \"originals\" folder contains the unmodified images.</p>
        <p>Send Ed a message if you see a photo that doesn't actually contain a bird!<br>Those images can be used to improve the AI model's accuracy.</p>
	<p>Here's a sample of the birds I saw (the numbers indicate my confidence, where 1 is the highest):</p>
        <img src=\"cid:{image_cid}\">
        <p>Hope you have a great day!</p>
        <p>Regards,<br>Ed's Birb Watcher</p>
    </body>
</html>
"
