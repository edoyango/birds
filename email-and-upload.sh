#!/bin/bash

set -ue

# save the date in yyyy-mm-dd format into the "date" variable
printf -v date '%(%Y-%m-%d)T' -1

# create a sample
python3 stack_sample_images.py observations/$date/triggers observations/$date/sample.jpg

# copy new files
rclone copy observations/ google-drive-birds:/

# create a public, read-only link to the folder
drive_link=$(rclone link google-drive-birds:/$date)

python3 send_birb_summary.py \
    -s "Birb watcher update" \
    -i observations/$date/sample.jpg \
    eds.birb.watcher@gmail.com \
    "Birb Watcher" \
    email-lists.csv \
    "<html>
    <body>
        <p>Hi {FIRST},</p>
        <p>You can view the birds I saw today via <a href=\"${drive_link}\">this google drive link</a>!</p>
        <p>The \"triggers\" folder shows what the AI model thought was a bird.<br>The \"originals\" folder contains the unmodified images.</p>
        <p>Send Ed a message if you see a photo that doesn't actually contain a bird!<br>Those images can be used to improve the AI model's accuracy.</p>
        <p>Here's a sample:</p>
        <img src=\"cid:{image_cid}\">
        <p>Hope you have a great day!</p>
        <p>Regards,<br>Ed's Birb Watcher</p>
    </body>
</html>
"
