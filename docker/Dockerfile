# ─────────────────────────────────────────────────────────────────────────────
# linuxai — Docker demo environment
# Base: ubuntu:22.04 (systemd-free, mimics a real Linux server)
# ─────────────────────────────────────────────────────────────────────────────

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── System packages ───────────────────────────────────────────────────────────
# Note: dmesg is part of util-linux (not a separate package on Ubuntu 22.04)
#       systemd is excluded — it's incompatible with minimal Docker containers
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        procps \
        coreutils \
        findutils \
        util-linux \
        systemctl \
        stress-ng \
        curl \
        iproute2 \
        net-tools \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir openai langchain langchain-core python-dotenv

# ── Copy project source ───────────────────────────────────────────────────────
WORKDIR /app
COPY tools.py        ./tools.py
COPY llm_client.py   ./llm_client.py
COPY agent.py        ./agent.py
COPY cli.py          ./cli.py
COPY safety.py       ./safety.py
COPY gap_log.py      ./gap_log.py
COPY orchestrator.py ./orchestrator.py

# ── Copy seed script ─────────────────────────────────────────────────────────
COPY docker/seed_problems.sh /usr/local/bin/seed_problems
RUN chmod +x /usr/local/bin/seed_problems

# ── Make linuxai runnable as a command ────────────────────────────────────────
RUN echo '#!/bin/sh\nexec python3 /app/cli.py "$@"' > /usr/local/bin/linuxai \
    && chmod +x /usr/local/bin/linuxai

CMD ["/bin/bash"]
