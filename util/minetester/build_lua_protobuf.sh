#! /bin/sh

cd lib/lua_protobuf

gcc -O2 -shared -fPIC -I "../lua/src" pb.c -o pb.so

