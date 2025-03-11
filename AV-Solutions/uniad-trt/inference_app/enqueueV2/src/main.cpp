/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <string.h>
#include <vector>
#include <fstream>
#include <numeric>
#include <cmath>
#include <sys/stat.h>
#include "uniad.hpp"
#include "visualize.hpp"
#include "tensor.hpp"
#include "pre_process.hpp"
#include "post_process.hpp"

void readBinFileToVec(const std::string& filename, void* vec, std::vector<int>& vec_shape, const std::vector<int>& ref_shape, const std::size_t vec_dsize) {
    std::ifstream bin_file(filename, std::ios::binary);
    if (!bin_file) {
        printf("[ERROR] Could not read file at: %s\n", filename.c_str());
        return;
    }
    bin_file.seekg(0, bin_file.end);
    int file_length = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    int dynamic_idx = -1;
    for (size_t shape_id=0; shape_id<vec_shape.size(); ++shape_id) {
        if (vec_shape[shape_id] < 0) dynamic_idx = shape_id;
    }
    if (dynamic_idx < 0) {
        if (file_length != vec_dsize*std::accumulate(vec_shape.begin(), vec_shape.end(), 1, std::multiplies<int>())) {
            printf("[ERROR] File size not match at: %s\n", filename.c_str());
            return;
        }
    } else {
        // dynamic shape
        vec_shape[dynamic_idx] = 1;
        int actual_shape = file_length / (vec_dsize*std::accumulate(vec_shape.begin(), vec_shape.end(), 1, std::multiplies<int>()));
        int error_ = file_length % (vec_dsize*std::accumulate(vec_shape.begin(), vec_shape.end(), 1, std::multiplies<int>()));
        if (error_ > 0) {
            printf("[ERROR] Invalid dynamic shape %d (residual %d) at position %d for %s\n", actual_shape, error_, dynamic_idx, filename.c_str());
            return;
        }
        if (actual_shape > ref_shape[dynamic_idx]) {
            printf("[ERROR] Invalid dynamic shape %d (maximum %d) at position %d for %s\n", actual_shape, ref_shape[dynamic_idx], dynamic_idx, filename.c_str());
            return;
        }
        vec_shape[dynamic_idx] = actual_shape;
    }
    bin_file.read((char *)vec, file_length);
    bin_file.close();
    return;
}

std::vector<std::vector<std::string>> parse_input_info(const std::string& input_pth, size_t num_inputs, int bias) {
    std::string file_name = input_pth;
    if (file_name[file_name.size()-1]!='/') file_name += "/";
    file_name += "info.txt";
    std::ifstream info_file(file_name);
    std::string info_str;
    std::vector<std::vector<std::string>> infos_vec;
    for (int _i=0; _i<bias; ++_i) std::getline(info_file, info_str);
    while (std::getline(info_file, info_str) && infos_vec.size()<num_inputs) {
        std::vector<std::string> info_;
        while (info_str.find(';') != std::string::npos) {
            size_t length = info_str.find(';');
            info_.push_back(info_str.substr(0, length));
            info_str = info_str.substr(length+1);
        }
        infos_vec.push_back(info_);
    }
    info_file.close();
    return infos_vec;
}

void load_input(const std::string& input_pth, std::vector<UniAD::KernelInput>& inputs, int num_inputs) {
    printf("[INFO] Loading input from %s\n", input_pth.c_str());
    std::vector<std::string> input_to_load{
        "timestamp", "l2g_r_mat", "l2g_t", "command",
        "img_metas_can_bus", "img_metas_lidar2img"
    };
    std::vector<float> curr_scene = std::vector<float>(32, 0);
    std::vector<int> scene_shape = std::vector<int>(1, 32);
    const UniAD::KernelParams kernel_params_ref;
    for (int file_id=0; file_id<num_inputs; ++file_id) {
        UniAD::KernelInput input_instance;
        std::string file_name = input_pth;
        if (file_name[file_name.size()-1] != '/') file_name += "/";
        for (std::string input_type : input_to_load) {
            if (input_instance.input_shapes.find(input_type) == input_instance.input_shapes.cend()) {
                // static shape
                input_instance.input_shapes[input_type] = kernel_params_ref.input_max_shapes.at(input_type);
            }
            readBinFileToVec(
                file_name+input_type+"/"+std::to_string(file_id)+".bin", 
                input_instance.data_ptrs[input_type],
                input_instance.input_shapes[input_type],
                kernel_params_ref.input_max_shapes.at(input_type),
                kernel_params_ref.input_sizes.at(input_type)
            );
        }
        std::vector<float> input_scene = std::vector<float>(32, 0);
        readBinFileToVec(
            file_name+"img_metas_scene_token/"+std::to_string(file_id)+".bin", 
            input_scene.data(),
            scene_shape,
            scene_shape,
            sizeof(float)
        );
        if (input_scene != curr_scene) {
            input_instance.use_prev_bev[0] = 0.;
            printf("[INFO] Scene changed at file id %d.\n", file_id);
            curr_scene = input_scene;
        } else input_instance.use_prev_bev[0] = 1.;
        input_instance.input_shapes["use_prev_bev"] = kernel_params_ref.input_max_shapes.at("use_prev_bev");
        inputs.push_back(input_instance);
    }
    return;
}

static void visualize(const std::vector<unsigned char*> images, const UniAD::KernelInput& input_instance, const UniAD::KernelOutput& output_instance, const std::string& save_path,
                      cudaStream_t stream) {
    std::vector<std::pair<float, float>> planning_traj = decode_planning_traj(output_instance);
    std::string command = decode_command(input_instance);
    std::vector<std::vector<float>> pred_bbox = decode_bbox(output_instance);

    int lidar_size = 1200;
    int content_width = lidar_size + 900;
    nv::ImageArtistParameter image_artist_param;
    image_artist_param.num_camera = images.size();
    image_artist_param.image_width = 1600;
    image_artist_param.image_height = 900;
    image_artist_param.image_stride = image_artist_param.image_width * 3;
    for (size_t i=0; i<input_instance.img_metas_lidar2img.size(); i+=4) {
        nvtype::Float4 transform_vec(input_instance.img_metas_lidar2img[i], input_instance.img_metas_lidar2img[i+1],
                                    input_instance.img_metas_lidar2img[i+2], input_instance.img_metas_lidar2img[i+3]);
        image_artist_param.viewport_nx4x4.push_back(transform_vec);
    }
    int gap = 0;
    int camera_width = content_width/3;
    int camera_height = static_cast<float>(camera_width / (float)image_artist_param.image_width * image_artist_param.image_height);
    int content_height = 2*camera_height + 3*content_width/4;
    nv::SceneArtistParameter scene_artist_param;
    scene_artist_param.width = content_width;
    scene_artist_param.height = content_height;
    scene_artist_param.stride = scene_artist_param.width * 3;

    nv::Tensor scene_device_image(std::vector<int>{scene_artist_param.height, scene_artist_param.width, 3}, nv::DataType::UInt8);
    scene_device_image.memset(0x00, stream);

    scene_artist_param.image_device = scene_device_image.ptr<unsigned char>();
    auto scene = nv::create_scene_artist(scene_artist_param);

    nv::BEVArtistParameter bev_artist_param;
    bev_artist_param.image_width = content_width;
    bev_artist_param.image_height = content_height;
    bev_artist_param.rotate_x = 70.0f;
    bev_artist_param.norm_size = lidar_size * 0.5f;
    bev_artist_param.cx = content_width * 0.5f;
    bev_artist_param.cy = content_height * 0.5f + camera_height;
    bev_artist_param.image_stride = scene_artist_param.stride;

    auto bev_visualizer = nv::create_bev_artist(bev_artist_param);
    bev_visualizer->draw_ego();
    for (int r=15; r<=60; r+=15) bev_visualizer->draw_circle(0, 0, r);
    bev_visualizer->draw_planning_traj(planning_traj, command);
    bev_visualizer->draw_prediction(pred_bbox, true);
    bev_visualizer->apply(scene_device_image.ptr<unsigned char>(), stream);

    int offset_cameras[][3] = {
        {camera_width, 0, 0},
        {camera_width*2, 0, 0},
        {0, 0, 0},
        {camera_width, camera_height, 1},
        {0, camera_height, 1},
        {camera_width*2, camera_height, 1}};

    auto visualizer = nv::create_image_artist(image_artist_param);
    for (size_t icamera = 0; icamera < images.size(); ++icamera) {
        int ox = offset_cameras[icamera][0];
        int oy = offset_cameras[icamera][1];
        bool xflip = static_cast<bool>(offset_cameras[icamera][2]);
        visualizer->draw_prediction(icamera, pred_bbox, xflip);
        visualizer->draw_planning_traj(icamera, planning_traj, xflip);

        nv::Tensor device_image(std::vector<int>{900, 1600, 3}, nv::DataType::UInt8);
        device_image.copy_from_host(images[icamera], stream);

        if (xflip) {
            auto clone = device_image.clone(stream);
            scene->flipx(clone.ptr<unsigned char>(), clone.size(1), clone.size(1) * 3, clone.size(0), device_image.ptr<unsigned char>(),
                        device_image.size(1) * 3, stream);
            checkRuntime(cudaStreamSynchronize(stream));
        }
        visualizer->apply(device_image.ptr<unsigned char>(), stream);

        scene->resize_to(device_image.ptr<unsigned char>(), ox, oy, ox + camera_width, oy + camera_height, device_image.size(1),
                        device_image.size(1) * 3, device_image.size(0), 0.8f, stream);
        checkRuntime(cudaStreamSynchronize(stream));
    }

    stbi_write_jpg(save_path.c_str(), scene_device_image.size(1), scene_device_image.size(0), 3,
                    scene_device_image.to_host(stream).ptr(), 100);
}

std::shared_ptr<UniAD::Kernel> create_kernel(const std::string& engine_pth) {
    UniAD::KernelParams params;
    params.trt_engine = engine_pth;
    std::shared_ptr<UniAD::Kernel> instance = std::make_shared<UniAD::KernelImplement>();
    instance->init(params);
    return instance;
}

void temporal_info_assign(UniAD::KernelInput& input_t, const UniAD::KernelOutput& output_t_1) {
    input_t.prev_track_intances0.assign(output_t_1.prev_track_intances0_out.begin(), output_t_1.prev_track_intances0_out.end());
    input_t.data_ptrs["prev_track_intances0"] = input_t.prev_track_intances0.data();
    input_t.input_shapes["prev_track_intances0"] = output_t_1.output_shapes.at("prev_track_intances0_out");

    input_t.prev_track_intances1.assign(output_t_1.prev_track_intances1_out.begin(), output_t_1.prev_track_intances1_out.end());
    input_t.data_ptrs["prev_track_intances1"] = input_t.prev_track_intances1.data();
    input_t.input_shapes["prev_track_intances1"] = output_t_1.output_shapes.at("prev_track_intances1_out");

    input_t.prev_track_intances3.assign(output_t_1.prev_track_intances3_out.begin(), output_t_1.prev_track_intances3_out.end());
    input_t.data_ptrs["prev_track_intances3"] = input_t.prev_track_intances3.data();
    input_t.input_shapes["prev_track_intances3"] = output_t_1.output_shapes.at("prev_track_intances3_out");

    input_t.prev_track_intances4.assign(output_t_1.prev_track_intances4_out.begin(), output_t_1.prev_track_intances4_out.end());
    input_t.data_ptrs["prev_track_intances4"] = input_t.prev_track_intances4.data();
    input_t.input_shapes["prev_track_intances4"] = output_t_1.output_shapes.at("prev_track_intances4_out");

    input_t.prev_track_intances5.assign(output_t_1.prev_track_intances5_out.begin(), output_t_1.prev_track_intances5_out.end());
    input_t.data_ptrs["prev_track_intances5"] = input_t.prev_track_intances5.data();
    input_t.input_shapes["prev_track_intances5"] = output_t_1.output_shapes.at("prev_track_intances5_out");

    input_t.prev_track_intances6.assign(output_t_1.prev_track_intances6_out.begin(), output_t_1.prev_track_intances6_out.end());
    input_t.data_ptrs["prev_track_intances6"] = input_t.prev_track_intances6.data();
    input_t.input_shapes["prev_track_intances6"] = output_t_1.output_shapes.at("prev_track_intances6_out");

    input_t.prev_track_intances8.assign(output_t_1.prev_track_intances8_out.begin(), output_t_1.prev_track_intances8_out.end());
    input_t.data_ptrs["prev_track_intances8"] = input_t.prev_track_intances8.data();
    input_t.input_shapes["prev_track_intances8"] = output_t_1.output_shapes.at("prev_track_intances8_out");

    input_t.prev_track_intances9.assign(output_t_1.prev_track_intances9_out.begin(), output_t_1.prev_track_intances9_out.end());
    input_t.data_ptrs["prev_track_intances9"] = input_t.prev_track_intances9.data();
    input_t.input_shapes["prev_track_intances9"] = output_t_1.output_shapes.at("prev_track_intances9_out");

    input_t.prev_track_intances11.assign(output_t_1.prev_track_intances11_out.begin(), output_t_1.prev_track_intances11_out.end());
    input_t.data_ptrs["prev_track_intances11"] = input_t.prev_track_intances11.data();
    input_t.input_shapes["prev_track_intances11"] = output_t_1.output_shapes.at("prev_track_intances11_out");

    input_t.prev_track_intances12.assign(output_t_1.prev_track_intances12_out.begin(), output_t_1.prev_track_intances12_out.end());
    input_t.data_ptrs["prev_track_intances12"] = input_t.prev_track_intances12.data();
    input_t.input_shapes["prev_track_intances12"] = output_t_1.output_shapes.at("prev_track_intances12_out");

    input_t.prev_track_intances13.assign(output_t_1.prev_track_intances13_out.begin(), output_t_1.prev_track_intances13_out.end());
    input_t.data_ptrs["prev_track_intances13"] = input_t.prev_track_intances13.data();
    input_t.input_shapes["prev_track_intances13"] = output_t_1.output_shapes.at("prev_track_intances13_out");

    input_t.prev_timestamp.assign(output_t_1.prev_timestamp_out.begin(), output_t_1.prev_timestamp_out.end());
    input_t.data_ptrs["prev_timestamp"] = input_t.prev_timestamp.data();
    input_t.input_shapes["prev_timestamp"] = output_t_1.output_shapes.at("prev_timestamp_out");

    input_t.prev_l2g_r_mat.assign(output_t_1.prev_l2g_r_mat_out.begin(), output_t_1.prev_l2g_r_mat_out.end());
    input_t.data_ptrs["prev_l2g_r_mat"] = input_t.prev_l2g_r_mat.data();
    input_t.input_shapes["prev_l2g_r_mat"] = output_t_1.output_shapes.at("prev_l2g_r_mat_out");

    input_t.prev_l2g_t.assign(output_t_1.prev_l2g_t_out.begin(), output_t_1.prev_l2g_t_out.end());
    input_t.data_ptrs["prev_l2g_t"] = input_t.prev_l2g_t.data();
    input_t.input_shapes["prev_l2g_t"] = output_t_1.output_shapes.at("prev_l2g_t_out");

    input_t.prev_bev.assign(output_t_1.bev_embed.begin(), output_t_1.bev_embed.end());
    input_t.data_ptrs["prev_bev"] = input_t.prev_bev.data();
    input_t.input_shapes["prev_bev"] = output_t_1.output_shapes.at("bev_embed");

    input_t.max_obj_id.assign(output_t_1.max_obj_id_out.begin(), output_t_1.max_obj_id_out.end());
    input_t.data_ptrs["max_obj_id"] = input_t.max_obj_id.data();
    input_t.input_shapes["max_obj_id"] = output_t_1.output_shapes.at("max_obj_id_out");
    return;
}

int main(int argc, char** argv) {
    assert (argc > 5);
    const std::string engine_pth = argv[1];
    const std::string plugin_pth = argv[2];
    const std::string input_pth = argv[3];
    const std::string output_pth = argv[4];
    const int num_frames = std::stoi(argv[5]);
    int num_warmup_iter=10;

    void* so_handle = dlopen(plugin_pth.c_str(), RTLD_NOW);
    // create the inference kernel
    std::shared_ptr<UniAD::Kernel> kernel = create_kernel(engine_pth);
    if (kernel == nullptr) {
        printf("[ERROR] Failed to create kernel in the main interface.\n");
        return -1;
    }
    kernel->print_info();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // prepare input, load input to Host memory
    std::vector<UniAD::KernelInput> inputs;
    load_input(input_pth, inputs, num_frames);

    std::vector<std::vector<std::string>> infos = parse_input_info(input_pth, num_frames, 0);
    const UniAD::KernelParams kernel_params_ref;
    int original_width, original_height, original_channel;
    auto images_dummy = load_images(infos, 0, original_width, original_height, original_channel);
    free_images(images_dummy);
    std::shared_ptr<ImgPreProcess> pre_processor = std::make_shared<ImgPreProcess>(original_width, original_height, original_channel, 0.25, 32);
    for (int imageid=0; imageid<num_frames; ++imageid) {
        auto images = load_images(infos, imageid);
        inputs[imageid].img = pre_processor->img_pre_processing(images, stream);
        free_images(images);
        inputs[imageid].input_shapes["img"] = kernel_params_ref.input_max_shapes.at("img");
    }
    // re-link the input
    for (size_t input_id=0; input_id<inputs.size(); ++input_id) {
        inputs[input_id].data_ptrs["prev_track_intances0"] = inputs[input_id].prev_track_intances0.data();
        inputs[input_id].data_ptrs["prev_track_intances1"] = inputs[input_id].prev_track_intances1.data();
        inputs[input_id].data_ptrs["prev_track_intances3"] = inputs[input_id].prev_track_intances3.data();
        inputs[input_id].data_ptrs["prev_track_intances4"] = inputs[input_id].prev_track_intances4.data();
        inputs[input_id].data_ptrs["prev_track_intances5"] = inputs[input_id].prev_track_intances5.data();
        inputs[input_id].data_ptrs["prev_track_intances6"] = inputs[input_id].prev_track_intances6.data();
        inputs[input_id].data_ptrs["prev_track_intances8"] = inputs[input_id].prev_track_intances8.data();
        inputs[input_id].data_ptrs["prev_track_intances9"] = inputs[input_id].prev_track_intances9.data();
        inputs[input_id].data_ptrs["prev_track_intances11"] = inputs[input_id].prev_track_intances11.data();
        inputs[input_id].data_ptrs["prev_track_intances12"] = inputs[input_id].prev_track_intances12.data();
        inputs[input_id].data_ptrs["prev_track_intances13"] = inputs[input_id].prev_track_intances13.data();
        inputs[input_id].data_ptrs["prev_timestamp"] = inputs[input_id].prev_timestamp.data();
        inputs[input_id].data_ptrs["prev_l2g_r_mat"] = inputs[input_id].prev_l2g_r_mat.data();
        inputs[input_id].data_ptrs["prev_l2g_t"] = inputs[input_id].prev_l2g_t.data();
        inputs[input_id].data_ptrs["prev_bev"] = inputs[input_id].prev_bev.data();
        inputs[input_id].data_ptrs["timestamp"] = inputs[input_id].timestamp.data();
        inputs[input_id].data_ptrs["l2g_r_mat"] = inputs[input_id].l2g_r_mat.data();
        inputs[input_id].data_ptrs["l2g_t"] = inputs[input_id].l2g_t.data();
        inputs[input_id].data_ptrs["img"] = inputs[input_id].img.data();
        inputs[input_id].data_ptrs["img_metas_can_bus"] = inputs[input_id].img_metas_can_bus.data();
        inputs[input_id].data_ptrs["img_metas_lidar2img"] = inputs[input_id].img_metas_lidar2img.data();
        inputs[input_id].data_ptrs["command"] = inputs[input_id].command.data();
        inputs[input_id].data_ptrs["use_prev_bev"] = inputs[input_id].use_prev_bev.data();
        inputs[input_id].data_ptrs["max_obj_id"] = inputs[input_id].max_obj_id.data();
    }

    // warmup
    UniAD::KernelOutput dummy_output;
    printf("[INFO] engine wram-up start\n");
    for (int i=0; i<num_warmup_iter; ++i) {
        cudaStreamSynchronize(stream);
        kernel->forward_one_frame(inputs[0], dummy_output, false, stream);
        cudaStreamSynchronize(stream);
    }
    printf("[INFO] engine wram-up done\n");

    // test inference latency
    std::vector<UniAD::KernelOutput> outputs;
    for (int i=0; i<num_frames; ++i) {
        printf("[INFO] Inferencing frame %d.\n", i);
        UniAD::KernelOutput _output;
        if (i > 0) temporal_info_assign(inputs[i], outputs[outputs.size()-1]);
        cudaStreamSynchronize(stream);
        kernel->forward_one_frame(inputs[i], _output, true, stream);
        cudaStreamSynchronize(stream);
        outputs.push_back(_output);
    }
    cudaStreamSynchronize(stream);

    // visualization
    struct stat sb;
    std::string img_dump_path = output_pth;
    if (img_dump_path[img_dump_path.size()-1]!='/') img_dump_path += "/";
    img_dump_path += "dumped_video_results/";
    if (stat(img_dump_path.c_str(), &sb) != 0) mkdir(img_dump_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    for (int imageid=0; imageid<num_frames; ++imageid) {
        auto images_vis = load_images(infos, imageid);
        visualize(images_vis, inputs[imageid], outputs[imageid], img_dump_path+std::to_string(imageid)+".jpg", stream);
        free_images(images_vis);
    }
    printf("[INFO] Visualization results have been dumped to %s.\n", img_dump_path.c_str());
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    // dlclose(so_handle);
    return 0;
}