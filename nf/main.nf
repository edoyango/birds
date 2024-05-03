include {RCLONE_LSF} from "./modules/rclone.nf"
include {RCLONE_COPY_DOWN} from "./modules/rclone.nf"
include {YOLO} from "./modules/yolo.nf"
include {FFMPEG_CRF} from "./modules/ffmpeg.nf"
include {MP42GIF} from "./modules/ffmpeg.nf"
include {RCLONE_COPY_UP as RCLONE_UPLOAD_TRIGGERS} from "./modules/rclone.nf"
include {RCLONE_COPY_UP as RCLONE_UPLOAD_ORIGINALS} from "./modules/rclone.nf"
include {RCLONE_COPY_UP as RCLONE_UPLOAD_INSTANCES} from "./modules/rclone.nf"
include {RCLONE_COPY_UP as RCLONE_UPLOAD_GIF} from "./modules/rclone.nf"

process ORGANISE {

    cpus 8
    memory "8 GB"
    container "linuxserver/ffmpeg"

    input:
    path(encoded_videos)
    path(original_trigger_images)
    path(instances)

    output:
    path("triggers", type: "dir")
    path("originals", type: "dir")
    path("instances-*.tar", type: "file")

    shell:
    '''
    mkdir triggers originals instances
    cd triggers
    cp -s ../trigger-*.mp4 . &
    tar --dereference -cf jpgs.tar ../trigger-*.jpg &
    cd ../originals
    cp -s ../original-*.mp4 . &
    tar --dereference -cf jpgs.tar ../original-*.jpg &
    cd ..
    find -L -name 'instances-*' -maxdepth 1 -type d -exec tar --dereference -cf {}.tar {} \\; &
    ffmpeg -y -framerate 1/2 -pattern_type glob -i 'trigger-*.jpg' -c:v libx265 -x265-params pools=!{task.cpus} -crf 28 triggers/jpgs.mp4
    ffmpeg -y -framerate 1/2 -pattern_type glob -i 'original-*.jpg' -c:v libx265 -x265-params pools=!{task.cpus} -crf 28 originals/jpgs.mp4
    wait
    '''
}

process GIFSICLE {

    cpus 1
    memory "1 GB"
    container "edoyango/gifsicle:1.94"

    input:
    path(gif)

    output:
    path("sample-${gif.baseName}-optimized.gif")

    script:
    """
    gifsicle -O3 --lossy=35 -i "${gif}" -o "sample-${gif.baseName}-optimized.gif"
    """
}

process EMAIL {

    cpus 1
    memory "1 GB"
    executor "local"

    input:
    each date
    path(gif)
    path(instances)
    path(email_list)
    each rclone_prefix

    shell:
    '''
    # create a google drive link
    gdrive_link=$(rclone link "!{rclone_prefix}/!{date}/triggers")
    find -L -maxdepth 1 -name 'instances-*.tar' -exec tar -tf {} \\; > instances_tar_contents.txt
    nbirbs=$(grep .jpg instances_tar_contents.txt | wc -l)
    birblist=$(\\
    for species in $(grep -v .jpg instances_tar_contents.txt | awk -F/ '{ if (NF>3) print $3 }' | sort | uniq); \\
    do \\
        echo "<li>${species}: $(awk -F/ -v species=${species} '$3==species{ print $4 }' instances_tar_contents.txt | wc -l)</li>"; \\
    done)

    send_birb_summary.py \\
        -s "Birb watcher update" \\
        -i "!{gif}" \\
        eds.birb.watcher@gmail.com \\
        "Birb Watcher" \\
        "!{email_list}" \\
        "<html>
        <body>
            <p>Hi {FIRST},</p>
            <p>I saw birds ${nbirbs} times today! These were:</p>
            <ul>
            $birblist
            </ul>
            <p>You can view all the recordings of birds I saw today via <a href=\"${gdrive_link}\">this google drive link</a>!</p>
            <p>Send Ed a message if you see that the model has incorrectly identified a bird!<br>Those images can be used to improve the AI model's accuracy.</p>
        <p>Here's one of the videos with birds:</p>
            <img src=\"cid:{image_cid}\">
            <p>Hope you have a great day!</p>
            <p>Regards,<br>Ed's Birb Watcher</p>
        </body>
        </html>"
    '''
}

workflow birbs_processing {

    ch_date = channel.of(params.date)
    ch_rclone_prefix = channel.of(params.rclone_prefix)

    RCLONE_LSF(ch_date, ch_rclone_prefix)
        .splitCsv(header: true)
        .map{row -> params.rclone_prefix + "/" + params.date + "/" + row.file}
        | RCLONE_COPY_DOWN
        | set{ch_videos}
    
    ch_model_detect = channel.fromPath(params.model_detect, type: "file", checkIfExists: true)
    ch_model_cls = channel.fromPath(params.model_cls, type: "file", checkIfExists: true)

    (ch_videos, ch_model_detect, ch_model_cls) = ch_videos
        .combine(ch_model_detect.collect())
        .combine(ch_model_cls.collect())
        .multiMap{it ->
            video: it[0]
            model_detect: it[1]
            model_cls: it[2]
        }

    (ch_yolo_videos, ch_yolo_imgs, ch_yolo_instances) = YOLO(ch_videos, ch_model_detect, ch_model_cls)
    ch_videos_encoded = FFMPEG_CRF(ch_yolo_videos.flatten())
    ch_yolo_instances.collect().count().view()
    (ch_triggers, ch_originals, ch_instances) = ORGANISE(
        ch_videos_encoded.collect(), 
        ch_yolo_imgs.flatten().collect(), 
        ch_yolo_instances.collect()
    )

    ch_gif = MP42GIF(ch_triggers)
        | GIFSICLE
    
    RCLONE_UPLOAD_GIF(ch_date, ch_rclone_prefix, ch_gif)
    RCLONE_UPLOAD_TRIGGERS(ch_date, ch_rclone_prefix, ch_triggers)
    RCLONE_UPLOAD_ORIGINALS(ch_date, ch_rclone_prefix, ch_originals)
    RCLONE_UPLOAD_INSTANCES(ch_date, ch_rclone_prefix, ch_instances)

    ch_email_list = channel.fromPath(params.email_list, checkIfExists: true)
    EMAIL(ch_date, ch_gif, ch_instances, ch_email_list, ch_rclone_prefix)
        
}

workflow {
    birbs_processing()
}
