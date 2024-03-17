#!/bin/bash

set -ue

# copy new files
rclone copy observations/ google-drive-birds:/

# save the date in yyyy-mm-dd format into the "date" variable
printf -v date '%(%Y-%m-%d)T\n' -1

# create a public, read-only link to the folder
drive_link=$(rclone link google-drive-birds:/$date)

python3 send_email_from_csv.py \
    -s "Birb watcher update" \
    eds.birb.watcher@gmail.com \
    "Birb Watcher" \
    email-lists.csv \
    "Hi {FIRST},

You can view the birds I saw today via the below google drive link:

${drive_link}

The \"triggers\" folder shows what the AI model thought was a bird.
The \"originals\" folder contains the unmodified images.

Send Ed a message if you see a photo that doesn't actually contain a bird!
Those images can be used to improve the AI model's accuracy.

Hope you have a great day!

Regards,
Ed's Birb Watcher"