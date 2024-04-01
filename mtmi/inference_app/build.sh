BUILD_PATH="build"
TARGET="x86_64"

mkdir -p ${BUILD_PATH}
mkdir -p ${BUILD_PATH}/${TARGET}
cd ${BUILD_PATH}/${TARGET}
cmake ../.. -DTARGET=${TARGET}
make
cd -    
