process FFMPEG_CRF {

    cpus 16
    memory "3 GB"
    container "linuxserver/ffmpeg"

    input:
    path(video)

    output:
    path("${video.baseName}.mp4")

    script:
    """
    ffmpeg -y -i "${video}" -c:v libx265 -x265-params pools=8 -crf 24 -preset slow "${video.baseName}.mp4"
    """
}

process MP42GIF {

    cpus 1 
    memory "2 GB"
    container  "linuxserver/ffmpeg"

    input:
    path("triggers")
    path("meta.csv")

    output:
    path("*.gif")

    shell:
    '''
    # get file with highest average instances
    file2gif=$(tail -n +2 -q meta.csv  | sort -n -t , -k 7 -r | head -n 1 | cut -d , -f 5)
    file2gif=${file2gif%.*}.mp4

    fname="$(basename ${file2gif})"
    fname="${fname%.*}"
    ffmpeg -y -t 40 -i "$file2gif" \\
        -vf "scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse" \\
        "$fname.gif"
    '''
}
