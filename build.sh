#! /bin/bash
#
# build.sh
# Distributed under terms of the MIT license.
#
# x86_64 aarch64 opencl
echo "---| number | target type|"
echo "-->   [1]   |   x86_64   |"
echo "-->   [2]   |   aarch64  |"
echo "-->   [3]   |   opencl   |"
echo "-->   [4]   |   all      |"
echo -n "Enter target type number: "
read n

x86_64_func(){
    python3 python/rpc_deploy.py x86_64
}
        
aarch64_func(){
    python3 python/rpc_deploy.py aarch64
}
        
opencl_func(){
    python3 python/rpc_deploy.py opencl
}


case $n in
1)
    echo "x86_64 build"
    x86_64_func
    ;;
2)
    echo "aarch64 build"
    aarch64_func
    ;;
3)
    echo "opencl build"
    opencl_func
    ;;
4)
    echo "all build"
    x86_64_func
    aarch64_func
    opencl_func
    ;;
*)
    echo "invalid"
    ;;
esac




