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

# Native C mypyc build of tokenizer_core and trie for fast test_max_nodes verification (kept separate from pure-Python sqlglot-src).
RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends gcc libc6-dev \
    && mkdir -p /app/sqlglot-c \
    && cp -r /app/sqlglot-src/sqlglot /app/sqlglot-c/sqlglot \
    && ln -s /app/sqlglot-src/tests /app/sqlglot-c/tests \
    && cd /app/sqlglot-c \
    && MYPYPATH=/app/sqlglot-c python -c "import os, sys; from setuptools import setup; from mypyc.build import mypycify; src = '/app/sqlglot-c/sqlglot'; files = [os.path.join(src, f) for f in ('tokenizer_core.py', 'trie.py')]; sys.argv = ['setup.py', 'build_ext', '--inplace', '-j', str(os.cpu_count() or 2)]; setup(name='sqlglotc_leaf', ext_modules=mypycify(files, opt_level='2', debug_level='0', separate=True))" \
    && rm -rf /app/sqlglot-c/build \
    && PYTHONPATH=/app/sqlglot-c python -c "import sqlglot, sqlglot.tokenizer_core as tc, sqlglot.trie as tr; assert tc.__file__.endswith('.so') and tr.__file__.endswith('.so')" \
    && chmod -R a+rX /app/sqlglot-c \
    && apt-get purge -y --auto-remove -qq gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Precompile TPC baseline and impact.marshal before copying the rest of blast_radius so rebuilds stay cached.
COPY blast_radius/__init__.py blast_radius/precompile_tpc.py /app/blast_radius/
RUN mkdir -p /app/map \
    && PYTHONPATH=/app python -m blast_radius.precompile_tpc

COPY map/ /app/map/
COPY blast_radius/impact_map.py blast_radius/precompile.py /app/blast_radius/
RUN PYTHONPATH=/app python -m blast_radius.precompile

COPY blast_radius/ /app/blast_radius/
# Which module's import runs each sqlglot line, and digests of what those imports build.
RUN PYTHONPATH=/app python -m blast_radius.import_state \
    && SKIP_INTEGRATION=0 PYTHONPATH=/app python -m blast_radius.precompile_cases \
    && python -c "import gzip; open('/app/map/values.marshal', 'wb').write(gzip.decompress(open('/app/map/values.bin', 'rb').read()))" \
    && python -m compileall -q /app/blast_radius \
    && chmod -R a+rX /app/blast_radius /app/map

ENV SKIP_INTEGRATION=0 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
CMD ["python", "-m", "blast_radius"]
