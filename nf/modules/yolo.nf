process YOLO {

    cpus 8
    memory "8 GB"
    container "ultralytics/ultralytics:8.2.79"
    conda "${moduleDir}/yolo-environment.yml"
    errorStrategy "ignore"
    time "1h"

    input:
    path(video)
    path(model_detect)
    path(model_cls)

    output:
    path("*-${video.baseName}/*.mkv"), optional: true // vidos
    path("*-${video.baseName}/first_frames/*.jpg"), optional: true // images
    path("instances-${video.baseName}", type: "dir") // instances images
    path("meta.csv"), optional: true

    script:
    """
    extract-birbs.py \\
        -m "${model_detect}" \\
        -o . \\
        -v "${video}" \\
        -c "${model_cls}"
    mv triggers triggers-${video.baseName}
    mv originals originals-${video.baseName}
    mv instances instances-${video.baseName}
    """
}
