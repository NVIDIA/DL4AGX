#############################################################################
# Copyright (c) 2018-2019 NVIDIA Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# File: DL4AGX/libs/opencv.BUILD
############################################################################
package(default_visibility = ["//visibility:public"])

config_setting(
    name = "aarch64_linux",
    values = { "crosstool_top": "//toolchains/D5L:aarch64-unknown-linux-gnu" }
)

config_setting(
    name = "aarch64_qnx",
    values = { "crosstool_top": "//toolchains/D5Q:aarch64-unknown-nto-qnx" }
)

################################ CORE ########################################
cc_import(
    name = "opencv_core_lib",
    shared_library = "lib/libopencv_core.so",
    #static_library = "lib/libopencv_core.a",
    visibility = ["//visibility:private"],
) 

cc_library(
    name = "opencv_core_headers",
    hdrs = glob([
        "include/opencv2/core.hpp",
        "include/opencv2/core/**/*.hpp",
        "include/opencv2/core.h",
        "include/opencv2/core/**/*.h"
    ]),
    includes = ["include/opencv2/core"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_core",
    deps = ["opencv_core_headers",
            "opencv_core_lib"],
)
################################ IMGCODECS ########################################
cc_import(
    name = "opencv_imgcodecs_lib",
    shared_library = "lib/libopencv_imgcodecs.so",
    #static_library = "lib/libopencv_imgcodecs.a",
    visibility = ["//visibility:private"],
) 

cc_library(
    name = "opencv_imgcodecs_headers",
    hdrs = glob([
        "include/opencv2/imgcodecs.hpp",
        "include/opencv2/imgcodecs/**/*.hpp",
        "include/opencv2/imgcodecs.h",
        "include/opencv2/imgcodecs/**/*.h"
    ]),
    includes = ["include/opencv2/imgcodecs"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_imgcodecs",
    deps = ["opencv_imgcodecs_headers", 
            "opencv_imgcodecs_lib"],
)
################################ IMGPROC ########################################
cc_import(
    name = "opencv_imgproc_lib",
    shared_library = "lib/libopencv_imgproc.so",
    #static_library = "lib/libopencv_imgproc.a",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_imgproc_headers",
    hdrs = glob([
        "include/opencv2/imgproc.hpp",
        "include/opencv2/imgproc/**/*.hpp",
        "include/opencv2/imgproc.h",
        "include/opencv2/imgproc/**/*.hpp"
    ]),
    includes = ["include/opencv2/imgproc"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_imgproc",
    deps = ["opencv_imgproc_headers",
            "opencv_imgproc_lib"],
)
################################ CALIB3D ########################################
cc_import(
    name = "opencv_calib3d_lib",
    shared_library = "lib/libopencv_calib3d.so",
    #static_library = "lib/libopencv_calib3d.a",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_calib3d_headers",
    hdrs = glob([
        "include/opencv2/calib3d.hpp",
        "include/opencv2/calib3d/**/*.hpp",
        "include/opencv2/calib3d.h",
        "include/opencv2/calib3d/**/*.h"
    ]),
    includes = ["include/opencv2/calib3d"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_calib3d",
    deps = ["opencv_calib3d_headers",
            "opencv_calib3d_lib"],
)
################################ FEATURES2D ########################################
cc_import(
    name = "opencv_features2d_lib",
    shared_library = "lib/libopencv_features2d.so",
    #static_library = "lib/libopencv_features2d.a",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_features2d_headers",
    hdrs = glob([
        "include/opencv2/features2d.hpp",
        "include/opencv2/features2d/**/*.hpp",
        "include/opencv2/features2d.h",
        "include/opencv2/features2d/**/*.h"
    ]),
    includes = ["include/opencv2/features2d"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_features2d",
    deps = ["opencv_features2d_headers",
            "opencv_features2d_lib"],
)
################################ FLANN ########################################
cc_import(
    name = "opencv_flann_lib",
    shared_library = "lib/libopencv_flann.so",
    #static_library = "lib/libopencv_flann.a",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_flann_headers",
    hdrs = glob([
        "include/opencv2/flann.hpp",
        "include/opencv2/flann/**/*.hpp",
        "include/opencv2/flann.h",
        "include/opencv2/flann/**/*.h"

    ]),
    includes = ["include/opencv2/flann"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_flann",
    deps = ["opencv_flann_headers",
          "opencv_flann_lib"],
)
################################ HIGHGUI ########################################
cc_import(
    name = "opencv_highgui_lib",
    shared_library = "lib/libopencv_highgui.so",
    #static_library = "lib/libopencv_highgui.a",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_highgui_headers",
    hdrs = glob([
        "include/opencv2/highgui.hpp",
        "include/opencv2/highgui/**/*.hpp",
        "include/opencv2/highgui.h",
        "include/opencv2/highgui/**/*.h"

    ]),
    includes = ["include/opencv2/highgui"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_highgui",
    deps = ["opencv_highgui_headers",
          "opencv_highgui_lib"],
)
################################ ML ########################################
cc_import(
    name = "opencv_ml_lib",
    shared_library = "lib/libopencv_ml.so",
    #static_library = "lib/libopencv_ml.a",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_ml_headers",
    hdrs = glob([
        "include/opencv2/ml.hpp",
        "include/opencv2/ml/**/*.hpp",
        "include/opencv2/ml.h",
        "include/opencv2/ml/**/*.h"
    ]),
    includes = ["include/opencv2/ml"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_ml",
    deps = ["opencv_ml_headers",
          "opencv_ml_lib"],
)
################################ OBJDETECT ########################################
cc_import(
    name = "opencv_objdetect_lib",
    shared_library = "lib/libopencv_objdetect.so",
    #static_library = "lib/libopencv_objdetect.a",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_objdetect_headers",
    hdrs = glob([
        "include/opencv2/objdetect.hpp",
        "include/opencv2/objdetect/**/*.hpp",
        "include/opencv2/objdetect.h",
        "include/opencv2/objdetect/**/*.h"
    ]),
    includes = ["include/opencv2/objdetect"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_objdetect",
    deps = ["opencv_objdetect_headers",
          "opencv_objdetect_lib"],
)
################################ PHOTO ########################################
cc_import(
    name = "opencv_photo_lib",
    shared_library = "lib/libopencv_photo.so",
    #static_library = "lib/libopencv_photo.a",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_photo_headers",
    hdrs = glob([
        "include/opencv2/photo.hpp",
        "include/opencv2/photo/**/*.hpp",
        "include/opencv2/photo.h",
        "include/opencv2/photo/**/*.h"
    ]),
    includes = ["include/opencv2/photo"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_photo",
    deps = ["opencv_photo_headers",
          "opencv_photo_lib"],
)
################################ SHAPE ########################################
cc_import(
    name = "opencv_shape_lib",
    shared_library = "lib/libopencv_shape.so",
    #static_library = "lib/libopencv_shape.a",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_shape_headers",
    hdrs = glob([
        "include/opencv2/shape.hpp",
        "include/opencv2/shape/**/*.hpp",
        "include/opencv2/shape.h",
        "include/opencv2/shape/**/*.h"
    ]),
    includes = ["include/opencv2/shape"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_shape",
    deps = ["opencv_shape_headers",
          "opencv_shape_lib"],
)
################################ STITCHING ########################################
cc_import(
    name = "opencv_stitching_lib",
    shared_library = "lib/libopencv_stitching.so",
    #static_library = "lib/libopencv_stitching.a",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_stitching_headers",
    hdrs = glob([
        "include/opencv2/stitching.hpp",
        "include/opencv2/stitching/**/*.hpp",
        "include/opencv2/stitching.h",
        "include/opencv2/stitching/**/*.h"
    ]),
    includes = ["include/opencv2/stitching"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_stitching",
    deps = ["opencv_stitching_headers",
          "opencv_stitching_lib"],
)
################################ SUPERRES ########################################
cc_import(
    name = "opencv_superres_lib",
    shared_library = "lib/libopencv_superres.so",
    #static_library = "lib/libopencv_superres.a",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_superres_headers",
    hdrs = glob([
        "include/opencv2/superres.hpp",
        "include/opencv2/superres/**/*.hpp",
        "include/opencv2/superres.h",
        "include/opencv2/superres/**/*.h"
    ]),
    includes = ["include/opencv2/superres"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_superres",
    deps = ["opencv_superres_headers",
          "opencv_superres_lib"],
)
################################ VIDEO ########################################
cc_import(
    name = "opencv_video_lib",
    shared_library = "lib/libopencv_video.so",
    #static_library = "lib/libopencv_video.a",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_video_headers",
    hdrs = glob([
        "include/opencv2/video.hpp",
        "include/opencv2/video/**/*.hpp",
        "include/opencv2/video.h",
        "include/opencv2/video/**/*.h"
    ]),
    includes = ["include/opencv2/video"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_video",
    deps = ["opencv_video_headers",
          "opencv_video_lib"],
)

################################ VIDEOIO ########################################
cc_import(
    name = "opencv_videoio_lib",
    shared_library = "lib/libopencv_videoio.so",
    #static_library = "lib/libopencv_videoio.a",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_videoio_headers",
    hdrs = glob([
        "include/opencv2/videoio.hpp",
        "include/opencv2/videoio/**/*.hpp",
        "include/opencv2/videoio.h",
        "include/opencv2/videoio/**/*.h"
    ]),
    includes = ["include/opencv2/videoio"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_videoio",
    deps = ["opencv_videoio_headers",
          "opencv_videoio_lib"],
)

################################ VIDEOSTAB ########################################
cc_import(
    name = "opencv_videostab_lib",
    shared_library = "lib/libopencv_videostab.so",
    #static_library = "lib/libopencv_videostab.a",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_videostab_headers",
    hdrs = glob([
        "include/opencv2/videostab.hpp",
        "include/opencv2/videostab/**/*.hpp",
        "include/opencv2/videostab.h",
        "include/opencv2/videostab/**/*.h",
    ]),
    includes = ["include/opencv/videostab"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_videostab",
    deps = ["opencv_videostab_headers",
          "opencv_videostab_lib"],
)

################################ VIDEOSTAB ########################################

cc_import(
    name = "opencv_gpu_x86_64_lib",
    shared_library = "lib/x86_64-linux-gnu/libopencv_gpu.so",
    #static_library = "lib/x86_64-linux-gnu/libopencv_gpu.a",
    visibility = ["//visibility:private"],
) 

cc_library(
    name = "opencv_gpu_headers",
    hdrs = glob([
        "include/opencv2/gpu.hpp",
        "include/opencv2/gpu/**/*.hpp",
        "include/opencv2/gpu.h",
        "include/opencv2/gpu/**/*.h",
    ]),
    includes = ["include/opencv/gpu"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "opencv_gpu",
    deps = ["opencv_gpu_headers"]
    + select({":aarch64_linux":[],
              ":aarch64_qnx":[],
              "//conditions:default":[],
    })
)

################################ OPENCV ########################################

cc_library(
    name = "opencv",
    srcs = select({
        ":aarch64_linux": glob(["lib/libopencv*.so"]),
        ":aarch64_qnx": glob(["lib/libopencv*.so"]),
        "//conditions:default": glob(["lib/x86_64-linux-gnu/libopencv*.so"]),
    }),
    hdrs = glob([
        "include/opencv2/**/*.h",
        "include/opencv2/**/*.hpp"
    ]),
    includes = ["include/opencv2"]
)
