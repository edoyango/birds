process FFMPEG_CRF {

    cpus 12
    memory "3 GB"
    time "60m"
    container "linuxserver/ffmpeg"

    input:
    path(video)

    output:
    path("${video.baseName}.mp4")

    script:
    """
    ffmpeg -y -i "${video}" -c:v libx265 -x265-params pools=12 -crf 24 -preset slow "${video.baseName}.mp4"
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
    # get 5 files with highest average instances
    for file2gif in $(tail -n +2 -q meta.csv  | sort -n -t , -k 7 -r | head -n "!{params.nsamples}" | cut -d , -f 5)
    do
        file2gif=${file2gif%.*}.mp4

        fname="$(basename ${file2gif})"
        fname="${fname%.*}"
        duration=$(ffmpeg -i "$file2gif" 2>&1 | grep "Duration"| cut -d ' ' -f 4 | sed s/,// | sed 's@\\..*@@g' | awk '{ split($1, A, ":"); split(A[3], B, "."); print 3600*A[1] + 60*A[2] + B[1] }')
        let "starttime = $duration/2 - 11"
        [ "$starttime" -lt 0 ] && starttime=0
        ffmpeg -y -ss "$starttime" -t "!{params.sample_duration}" -i "$file2gif" \\
            -vf "scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse" \\
            "$fname.gif"
    done
    '''
}
