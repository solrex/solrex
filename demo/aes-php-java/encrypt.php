<?php
$string = '{"a"="b"}';
$key128=pack('H*', '25c506a9e4a0b3100d2d86b49b83cf9a');
$iv=pack('H*', '00000000000000000000000000000000');
$enc_str = mcrypt_encrypt(MCRYPT_RIJNDAEL_128, $key128, $string, MCRYPT_MODE_CBC, $iv);
echo $enc_str;
