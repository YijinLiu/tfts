#!/bin/bash

main() {
    local pwd_path=`pwd`

    [ ""`dirname $pwd_path` == "/home" ] || error "--workdir must be set to a path under /home/"

    local host_username=`basename $pwd_path`
    local host_uid=`stat --format=%u .`
    local host_gid=`stat --format=%g .`

    # We won't be able to recreate the user when restarting a container that was stopped
    if id -u $host_username >& /dev/null && [ `id -u $host_username` -eq "$host_uid" ] ; then
        info "$host_username already has id $host_uid -- assuming we are resuming"
    else
        info "pwd is $pwd_path, name is $host_username, will set uid to $host_uid"

        sudo groupadd --gid $host_gid $host_username
        sudo useradd --uid "$host_uid" --gid $host_gid --shell /bin/bash --groups sudo,video \
            "$host_username" || error "Failed to add user $host_username"
        info "$host_username is created"

        sudo bash -c "echo '$host_username ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers.d/camect"  ||
            error "Failed to set sudo w/o password for $host_username"
        info "$host_username is granted root"
    fi

    sudo -u "$host_username" -H /bin/bash
}

colors=1;
info() {
    if [ "$colors" -eq 1 ] ; then
        echo -e "\e[33mINFO: $1\e[39m"
    else
        echo "INFO: $1"
    fi
}

error() {
    if [ "$colors" -eq 1 ] ; then
        echo -e "\e[31mFATAL: $1\e[39m"
    else
        echo "FATAL: $1"
    fi
    exit 1
}

main
