BUILD_PATH="build"
TARGET="${1:-orin}"
 
mkdir -p ${BUILD_PATH}
mkdir -p ${BUILD_PATH}/${TARGET}
cd ${BUILD_PATH}/${TARGET}
cmake ../.. -DTARGET=${TARGET} -DTRTROOT=${TRTROOT}
make
cd -
