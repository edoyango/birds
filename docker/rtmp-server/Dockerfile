FROM buildpack-deps:bullseye-curl

RUN apt-get update && apt-get install --no-install-recommends nginx-common libnginx-mod-rtmp nginx -y && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY nginx.conf /tmp/
RUN cat /tmp/nginx.conf >> /etc/nginx/nginx.conf
RUN ln -sf /dev/stdout /var/log/nginx/access.log && ln -sf /dev/stderr /var/log/nginx/error.log
CMD ["nginx", "-g", "daemon off;"]
