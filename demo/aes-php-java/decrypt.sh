#!/bin/bash
key128="25c506a9e4a0b3100d2d86b49b83cf9a"
iv="00000000000000000000000000000000"
#wget --no-check-certificate https://cp01-rdqa-dev354.cp01.baidu.com:8088/b20/update -O data.enc
openssl enc -aes-128-cbc -d -in data.enc -out data.dec -K $key128 -iv $iv -nopad
cat ./data.dec
