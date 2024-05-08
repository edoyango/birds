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

    output:
    path("*.gif")

    shell:
    '''
    file2gif="$(\\
        find -L triggers \\
            -type f \\
            -name '*.mp4' \\
            -size -3000000c \\
            -printf '%s %p\\n' | \\
        sort -nr | \\
        head -n 1 | \\
        cut -d " " -f 2\\
    )"
    fname="$(basename ${file2gif})"
    fname="${fname%.*}"
    ffmpeg -y -i "$file2gif" \\
        -vf "scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse" \\
        "$fname.gif"
    '''
}
