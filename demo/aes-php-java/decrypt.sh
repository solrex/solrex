#!/bin/bash
key128="25c506a9e4a0b3100d2d86b49b83cf9a"
iv="00000000000000000000000000000000"
openssl enc -aes-128-cbc -d -in data.enc -out data.dec -K $key128 -iv $iv -nopad
cat ./data.dec
