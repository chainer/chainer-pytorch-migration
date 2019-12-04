#!/bin/bash -uex

perl -pi.bak -e 's|http://archive\.ubuntu\.com/ubuntu/|mirror://mirrors.ubuntu.com/mirrors.txt|g' /etc/apt/sources.list
apt update
apt -y install python3 python3-pip

pip3 install -U --pre chainer cupy-cuda100
pip3 install pytorch-ignite
pip3 install -e '.[test]'

python3 -m pytest tests/
