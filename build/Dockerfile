FROM ubuntu:22.04
ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8

RUN apt update && apt upgrade -y && apt install -qq -y --no-install-recommends \
        apt-utils bash-completion build-essential ca-certificates cmake curl git gpg-agent \
        locales patchelf pkg-config protobuf-compiler python3-dev python3-pip python3-setuptools \
        software-properties-common sudo tzdata unzip vim wget zip && \
    pip install -U --no-cache-dir pip setuptools && \
    apt autoremove -y python3-pip && \
    pip3 install -U --no-cache-dir Pillow matplotlib numba pandas python-dateutil && \
    ln -sf /usr/share/zoneinfo/US/Pacific /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    locale-gen en_US.UTF-8

ADD install_cuda.sh /tmp/build/
WORKDIR /tmp/build
RUN cd /tmp/build && ./install_cuda.sh && cd / && rm -rf /tmp/*

RUN pip3 install -U --no-cache-dir tensorflow==2.14.0 tensorflow_probability[jax]==0.22.0

COPY fix_user.sh /sbin/fix_user.sh
ENTRYPOINT ["/sbin/fix_user.sh", "/sbin/fix_user.sh"]

# docker run -ti --runtime=nvidia --shm-size=1g -e NVIDIA_VISIBLE_DEVICES=0 --cap-add=SYS_PTRACE \
#     --security-opt seccomp=unconfined -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY \
#     --device /dev/snd -v /home/$USER:/home/$USER --workdir /home/$USER --name=tfp_0_22_0 \
#     $USER/tfp:0.22.0-gpu

# docker run -ti --runtime=nvidia --shm-size=1g -e NVIDIA_VISIBLE_DEVICES=0 --cap-add=SYS_PTRACE \
#     --security-opt seccomp=unconfined -v /run/desktop/mnt/host/wslg/.X11-unix:/tmp/.X11-unix \
#     -v /run/desktop/mnt/host/wslg:/mnt/wslg -e DISPLAY=:0 -e WAYLAND_DISPLAY=wayland-0 \
#     -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir -e PULSE_SERVER=/mnt/wslg/PulseServer \
#     --workdir /home/$USER --name=tfp_0_22_0 $USER/tfp:0.22.0-gpu
