# Sandbox image for the KV-cache RL environment.
# Agent code runs as a non-root user with no network and a writable workdir.
FROM python:3.11-slim

# System deps kept minimal — torch wheel handles the rest.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. Agent code never runs as root.
RUN useradd -m -u 1000 -s /bin/bash agent

WORKDIR /workspace

# Pre-install deps so each episode doesn't pay the install cost.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# The episode files (model.py, judge.py, etc.) are bind-mounted at runtime,
# not baked into the image. This lets us update the env without rebuilding.

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER agent
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]