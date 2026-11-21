# The scorer builds this on linux/amd64 and runs it with no network. The
# digest is the image the organizers labeled every commit in.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# Match the exact packages used during official labeling for runtime test verification.
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock && rm /tmp/requirements.lock

# sqlglot at the pinned commit, precompiled to .pyc for fast runtime verification.
ADD https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz /tmp/sqlglot.tar.gz
RUN mkdir -p /app/sqlglot-src \
    && tar -xzf /tmp/sqlglot.tar.gz -C /app/sqlglot-src --strip-components=1 \
    && rm /tmp/sqlglot.tar.gz \
    && python -m compileall -q /app/sqlglot-src

COPY blast_radius/ /app/blast_radius/
COPY map/ /app/map/
RUN python -m compileall -q /app/blast_radius && chmod -R a+rX /app

ENV SKIP_INTEGRATION=0 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
CMD ["python", "-m", "blast_radius"]
