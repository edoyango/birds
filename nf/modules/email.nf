process EMAIL {

    cpus 1
    memory "1 GB"
    executor "local"

    input:
    path(mailing_list)
    path(videos_and_images)

    script:
    """
    ls *
    """
}