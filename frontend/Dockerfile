# AirType Frontend Dockerfile
FROM nginx:alpine

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy frontend files
COPY . /usr/share/nginx/html

# Remove default nginx files
RUN rm -f /usr/share/nginx/html/Dockerfile \
    /usr/share/nginx/html/nginx.conf

# Create non-root user
RUN adduser -D -u 1001 appuser && \
    chown -R appuser:appuser /usr/share/nginx/html && \
    chown -R appuser:appuser /var/cache/nginx && \
    chown -R appuser:appuser /var/log/nginx && \
    touch /var/run/nginx.pid && \
    chown appuser:appuser /var/run/nginx.pid

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:80/ || exit 1

# Run nginx
CMD ["nginx", "-g", "daemon off;"]
