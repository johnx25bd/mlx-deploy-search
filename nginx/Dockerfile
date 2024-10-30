FROM nginx:alpine

RUN mkdir -p /var/cache/nginx && \
    chown -R nginx:nginx /var/cache/nginx && \
    chmod -R 755 /var/cache/nginx

RUN mkdir -p /var/run/nginx && \
    chown -R nginx:nginx /var/run/nginx && \
    chmod -R 755 /var/run/nginx

COPY conf/nginx.conf /etc/nginx/conf.d/default.conf
RUN chown -R nginx:nginx /etc/nginx/conf.d/default.conf

RUN sed -i 's/user  nginx;//g' /etc/nginx/nginx.conf && \
    sed -i 's/^pid.*//g' /etc/nginx/nginx.conf

CMD ["nginx", "-g", "daemon off; pid /var/run/nginx/nginx.pid;"]

USER nginx

EXPOSE 8080