include {RCLONE_LSF} from "./modules/rclone.nf"
include {RCLONE_COPY_DOWN} from "./modules/rclone.nf"
include {YOLO} from "./modules/yolo.nf"
include {FFMPEG_CRF} from "./modules/ffmpeg.nf"
include {MP42GIF} from "./modules/ffmpeg.nf"
include {RCLONE_COPY_UP as RCLONE_UPLOAD_TRIGGERS} from "./modules/rclone.nf"
include {RCLONE_COPY_UP as RCLONE_UPLOAD_ORIGINALS} from "./modules/rclone.nf"
include {RCLONE_COPY_UP as RCLONE_UPLOAD_INSTANCES} from "./modules/rclone.nf"
include {RCLONE_COPY_UP as RCLONE_UPLOAD_GIF} from "./modules/rclone.nf"
include {EMAIL} from "./modules/email.nf"

process ORGANISE {

    cpus 1
    memory "1 GB"

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
    cp -s ../trigger-*.{mp4,jpg} .
    cd ../originals
    cp -s ../original-*.{mp4,jpg} .
    cd ..
    find -L -name 'instances-*' -maxdepth 1 -type d -exec tar --dereference -cf {}.tar {} \\;
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
        
}

workflow {
    birbs_processing()
}
