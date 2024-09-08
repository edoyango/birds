process RCLONE_LSF {

    cpus 1
    memory "2 GB"
    module "rclone"
    executor "local"

    input:
    val date // YYYY-MM-DD
    val rclone_prefix // e.g. google-drive:birds

    output:
    path("videos.csv")

    script:
    """
    echo file > videos.csv
    rclone lsf "${rclone_prefix}/${date}" | grep .mkv >> videos.csv
    """

}

process RCLONE_COPY_DOWN {

    cpus 1
    memory "2 GB"
    module "rclone"
    
    input:
    val remote_fpath

    output:
    path("*??-??-??.mkv"), emit: mkv

    script:
    """
    rclone copy "${remote_fpath}" .
    """
}

process RCLONE_COPY_UP {

    cpus 1
    memory "2 GB"
    module "rclone"

    input:
    each date
    each rclone_prefix
    path(results)

    script:
    """
    dir="\$(echo "${results}" | awk -F '-' '{print \$1}')"
    rclone mkdir "${rclone_prefix}/${date}/\${dir}"
    for item in ${results}
    do
        rclone -L copy -P \${item} "${rclone_prefix}/${date}/\${dir}"
    done
    """

}
