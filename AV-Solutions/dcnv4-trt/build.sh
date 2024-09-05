BUILD_PATH="build"
TARGET=$1

mkdir -p ${BUILD_PATH}
mkdir -p ${BUILD_PATH}/${TARGET}
cd ${BUILD_PATH}/${TARGET}
cmake ../.. -DTARGET=$TARGET ${@:2}
make VERBOSE=1
cd -    
