#!/usr/bin/env bash

# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

run_embed() {

    mkdir -p $(pwd)/volumes/milvus
    
    cat << EOF > embedEtcd.yaml
listen-client-urls: http://0.0.0.0:2379
advertise-client-urls: http://0.0.0.0:2379
quota-backend-bytes: 4294967296
auto-compaction-mode: revision
auto-compaction-retention: '1000'
EOF

    cat << EOF > user.yaml
# Extra config to override default milvus.yaml
EOF
    if [ ! -f "./embedEtcd.yaml" ]
    then
        echo "embedEtcd.yaml file does not exist. Please try to create it in the current directory."
        exit 1
    fi

    if [ ! -f "./user.yaml" ]
    then
        echo "user.yaml file does not exist. Please try to create it in the current directory."
        exit 1
    fi
    
    podman run -d \
        --name milvus-standalone-papermatch \
        --security-opt seccomp:unconfined \
        -e ETCD_USE_EMBED=true \
        -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
        -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
        -e COMMON_STORAGETYPE=local \
        -v $(pwd)/volumes/milvus:/var/lib/milvus:Z \
        -v $(pwd)/embedEtcd.yaml:/milvus/configs/embedEtcd.yaml:Z \
        -v $(pwd)/user.yaml:/milvus/configs/user.yaml:Z \
        -p 19530:19530 \
        -p 9091:9091 \
        -p 2379:2379 \
        --health-cmd="curl -f http://localhost:9091/healthz" \
        --health-interval=30s \
        --health-start-period=90s \
        --health-timeout=20s \
        --health-retries=3 \
        docker.io/milvusdb/milvus:v2.5.15 \
        milvus run standalone  1> /dev/null
}

wait_for_milvus_running() {
    echo "Wait for Milvus Starting..."
    while true
    do
        res=`podman ps|grep milvus-standalone|grep healthy|wc -l`
        if [ $res -eq 1 ]
        then
            echo "Start successfully."
            echo "To change the default Milvus configuration, add your settings to the user.yaml file and then restart the service."
            break
        fi
        sleep 1
    done
}

start() {
    res=`podman ps|grep milvus-standalone|grep healthy|wc -l`
    if [ $res -eq 1 ]
    then
        echo "Milvus is running."
        exit 0
    fi

    res=`podman ps -a|grep milvus-standalone|wc -l`
    if [ $res -eq 1 ]
    then
        podman start milvus-standalone 1> /dev/null
    else
        run_embed
    fi

    if [ $? -ne 0 ]
    then
        echo "Start failed."
        exit 1
    fi

    wait_for_milvus_running
}

stop() {
    podman stop milvus-standalone 1> /dev/null

    if [ $? -ne 0 ]
    then
        echo "Stop failed."
        exit 1
    fi
    echo "Stop successfully."

}

delete_container() {
    res=`podman ps|grep milvus-standalone|wc -l`
    if [ $res -eq 1 ]
    then
        echo "Please stop Milvus service before delete."
        exit 1
    fi
    podman rm milvus-standalone 1> /dev/null
    if [ $? -ne 0 ]
    then
        echo "Delete milvus container failed."
        exit 1
    fi
    echo "Delete milvus container successfully."
}

delete() {
    delete_container
    rm -rf $(pwd)/volumes
    rm -rf $(pwd)/embedEtcd.yaml
    rm -rf $(pwd)/user.yaml
    echo "Delete successfully."
}

upgrade() {
    read -p "Please confirm if you'd like to proceed with the upgrade. The default will be to the latest version. Confirm with 'y' for yes or 'n' for no. > " check
    if [ "$check" == "y" ] ||[ "$check" == "Y" ];then
        res=`podman ps -a|grep milvus-standalone|wc -l`
        if [ $res -eq 1 ]
        then
            stop
            delete_container
        fi

        curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed_latest.sh && \
        bash standalone_embed_latest.sh start 1> /dev/null && \
        echo "Upgrade successfully."
    else
        echo "Exit upgrade"
        exit 0
    fi
}

case $1 in
    restart)
        stop
        start
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    upgrade)
        upgrade
        ;;
    delete)
        delete
        ;;
    *)
        echo "please use bash standalone_embed.sh restart|start|stop|upgrade|delete"
        ;;
esac
