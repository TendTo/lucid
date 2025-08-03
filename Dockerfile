FROM ubuntu:22.04

LABEL authors="Oliver Schon, Ernesto Casablanca"
LABEL workspace="lucid"
EXPOSE 3661

# Initial setup
RUN apt-get update && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends curl software-properties-common gpg-agent && \
    apt-get autoremove -y && \
    apt-get clean -y

# Install bazel
ARG BAZELISK_VERSION=v1.26.0
ARG BAZELISK_DOWNLOAD_SHA=6539c12842ad76966f3d493e8f80d67caa84ec4a000e220d5459833c967c12bc
RUN curl -fSsL -o /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64 \
    && ([ "${BAZELISK_DOWNLOAD_SHA}" = "dev-mode" ] || echo "${BAZELISK_DOWNLOAD_SHA} /usr/local/bin/bazel" | sha256sum --check --status - ) \
    && chmod 0755 /usr/local/bin/bazel

# Install required packages
ARG APT_PACKAGES="git python3 python3-pip build-essential python3-dev libibex-dev libnlopt-cxx-dev"
RUN export DEBIAN_FRONTEND=noninteractive && \
    add-apt-repository ppa:dreal/dreal -y && \
    apt-get install -y --no-install-recommends ${APT_PACKAGES} && \
    apt-get autoremove -y && \
    apt-get clean -y

# Install gurobi
WORKDIR /opt
ENV GUROBI_VERSION=12.0.0
ENV GUROBI_HOME=/opt/gurobi/linux64
ENV LD_LIBRARY_PATH=/usr/local/lib:/opt/gurobi/linux64/lib
RUN curl -fSsL -o gurobi${GUROBI_VERSION}_linux64.tar.gz https://packages.gurobi.com/12.0/gurobi${GUROBI_VERSION}_linux64.tar.gz && \
    tar xvf gurobi${GUROBI_VERSION}_linux64.tar.gz && \
    rm gurobi${GUROBI_VERSION}_linux64.tar.gz && \
    mv gurobi* /opt/gurobi

WORKDIR /app
COPY . .

# Allow python to run as root
RUN sed 's/python.toolchain(/python.toolchain(\nignore_root_user_error = True,/g' MODULE.bazel -i

# Fetch dependencies. Useful to avoid downloading them every time
# RUN bazel fetch --config=opt //lucid

# Build lucid
# RUN bazel build --config=opt //lucid

# Install pylucid bindings and clean up bazel
RUN pip install --upgrade pip && \
    pip install --ignore-installed --no-cache-dir ".[plot,verification,gui]" && \
    bazel clean --expunge

ENTRYPOINT [ "pylucid" ]

# RUN bazel build --config=opt --//tools:enable_static_build=True //lucid

# FROM alpine:3.21.3

# LABEL authors="Oliver Schon, Ernesto Casablanca"
# LABEL workspace="lucid"


# WORKDIR /app

# COPY --from=build /app/bazel-bin/lucid /sbin

# ENTRYPOINT ["lucid"]
