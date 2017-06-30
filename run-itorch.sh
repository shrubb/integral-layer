#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
LUA_PATH="/home/e_burkov/.luarocks/share/lua/5.1/?.lua;/home/e_burkov/.luarocks/share/lua/5.1/?/init.lua;/media/hpc4_Raid/e_burkov/Libraries/Torch7/install/share/lua/5.1/?.lua;/media/hpc4_Raid/e_burkov/Libraries/Torch7/install/share/lua/5.1/?/init.lua;./?.lua" LUA_CPATH="/home/e_burkov/.luarocks/lib/lua/5.1/?.so;./?.so;/media/hpc4_Raid/e_burkov/Libraries/Torch7/install/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so" itorch notebook --ip=*
