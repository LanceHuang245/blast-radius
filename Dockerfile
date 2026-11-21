# The scorer builds this on linux/amd64 and runs it with no network. The
# digest is the image the organizers labeled every commit in.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# sqlglot at the pinned commit: the selector patches these sources in memory to
# see what each change touches. It never runs them.
ADD https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz /tmp/sqlglot.tar.gz
RUN mkdir -p /app/sqlglot-src \
    && tar -xzf /tmp/sqlglot.tar.gz -C /app/sqlglot-src --strip-components=1 \
    && rm /tmp/sqlglot.tar.gz

COPY blast_radius/ /app/blast_radius/
COPY map/ /app/map/
RUN python -m compileall -q /app/blast_radius && chmod -R a+rX /app

WORKDIR /app
CMD ["python", "-m", "blast_radius"]
