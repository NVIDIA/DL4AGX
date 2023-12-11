#include <cuda_runtime_api.h>
#include <NvInferRuntime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <chrono>
#include <cstring>

#include <thread>
#include <pthread.h>

#include "lodepng.h"
#include "cudla_context.h"
#include <cuda_fp16.h>
#include "half.h"

#include "kernels.h"


using namespace nvinfer1;

class Logger : public ILogger
{
public:
	void log(Severity severity, const char* msg) noexcept override
	{
		// Only print error messages
		if (severity == Severity::kERROR)
		{
			std::cerr << msg << std::endl;
		}
	}
};

Logger gLogger;

template<typename T>
inline void destroy(T* obj)
{
	if (obj)
	{
		obj->destroy();
	}
}

struct thead_pack {
	cudaStream_t stream;
	std::vector<std::string> ins_names;
	std::map<std::string, std::unique_ptr<IExecutionContext>> *contexts;
};

struct conversion_thead_pack {
	cudaStream_t stream;
	float* input;
	void* output;
	float scale;
	int size;
};


void *launch_kernel(void *tp)
{
	cudaStream_t &stream = ((struct thead_pack *) tp)->stream;
	std::vector<std::string> &ins_names = ((struct thead_pack *) tp)->ins_names;
	std::map<std::string, std::unique_ptr<IExecutionContext>> *cur_contexts = ((struct thead_pack *) tp)->contexts;
	for (const auto & instanceName : ins_names)
	{	
		(*cur_contexts)[instanceName]->enqueueV3(stream);
	}
	cudaStreamSynchronize(stream);
    return NULL;
}

void *launch_conversion(void *tp) {
	cudaStream_t &stream = ((struct conversion_thead_pack *) tp)->stream;
	int size = ((struct conversion_thead_pack *) tp)->size;
	float scale = ((struct conversion_thead_pack *) tp)->scale;
	float* input = ((struct conversion_thead_pack *) tp)->input;
	void* output = ((struct conversion_thead_pack *) tp)->output;
	convert_float_to_int8(input, (int8_t *)output, size, scale, stream);
	cudaStreamSynchronize(stream);
	return NULL;
}


int main(int argc, char* argv[])
{
	// Check command line arguments
	if (argc != 2)
	{
		std::cerr << "Usage: " << argv[0] << " config.yaml" << std::endl;
		return -1;
	}

	// Load the YAML file
	YAML::Node config = YAML::LoadFile(argv[1]);

	// Get the number of iterations from the YAML file
	int numIterations = config["iterations"].as<int>();

	std::string seg_dla_path = config["seg_dla_path"].as<std::string>();
	std::string depth_dla_path = config["depth_dla_path"].as<std::string>();
	std::cout<<seg_dla_path<<std::endl;
	std::cout<<depth_dla_path<<std::endl;
	std::vector<float> scales{0.0219308696687, 0.0277800317854, 0.036009144038, 0.120857991278};
	std::vector<int> feat_sizes{32*256*256, 64*128*128, 160*64*64, 256*32*32};

	// Create a TensorRT runtime object
	//std::unique_ptr<IRuntime, decltype(&destroy<IRuntime>)> runtime{createInferRuntime(gLogger), destroy<IRuntime>};
	auto runtime_deleter = [](IRuntime *runtime) { runtime->destroy(); };
	std::unique_ptr<IRuntime, decltype(runtime_deleter)> runtime{createInferRuntime(gLogger), runtime_deleter};

	// Create a vector to hold the engines
	//std::vector<std::unique_ptr<ICudaEngine>> engines;
	std::map<std::string, std::unique_ptr<ICudaEngine>> engines;

	// Create a vector to hold execution contexts for all engines
	//std::vector<std::unique_ptr<IExecutionContext>> contexts;
	std::map<std::string, std::unique_ptr<IExecutionContext>> contexts;

	// Create a vector to hold bindings
	//std::vector<std::vector<float *>> bindings;
	auto cuda_deleter = [](void *ptr) { cudaFree(ptr); };
	//std::map<std::string, std::map<std::string, std::unique_ptr<void, decltype(cuda_deleter)>>> bindings;
	std::map<std::string, std::map<std::string, void *>> bindings;

	std::map<int, std::tuple<cudaStream_t, cudaGraph_t, cudaGraphExec_t, std::vector<std::string>>> instanceGroups;
	//std::map<int, cudaStream_t> streams;
	std::map<std::string, std::tuple<std::string, std::vector<std::string>>> inputs;

	for (const auto & input : config["inputs"])
	{
		std::cout << "Loading input " << input["name"] << std::endl;
		for (const auto & entry : std::filesystem::directory_iterator(input["path"].as<std::string>()))
		{
			if (entry.path().extension() == input["extension"].as<std::string>())
			{
				if (!inputs.count(input["name"].as<std::string>()))
				{
					inputs.emplace(input["name"].as<std::string>(), std::make_tuple(input["type"].as<std::string>(), std::vector<std::string>()));
				}
				std::get<1>(inputs[input["name"].as<std::string>()]).emplace_back(entry.path());
			}
		}
	}
	std::cout << std::endl;

	// Deserialize each engine from the YAML file
	for (const auto & instance : config["instances"])
	{
		const auto & enginePath = instance["engine"].as<std::string>();
		const auto & instanceName = instance["name"].as<std::string>();
		const int core = instance["core"].as<int>();
		// Read the engine file
		std::cout << "Deserializing " << enginePath << " for " << instanceName << " : " << core << std::endl;

		std::ifstream engineFile(enginePath, std::ios::binary);
		if (!engineFile)
		{
			std::cerr << "Error opening engine file: " << enginePath << std::endl;
			return -1;
		}

		// Get the size of the engine file
		engineFile.seekg(0, engineFile.end);
		long int fsize = engineFile.tellg();
		engineFile.seekg(0, engineFile.beg);

		// Read the engine file into a buffer
		std::vector<char> engineData(fsize);

		engineFile.read(engineData.data(), fsize);

		// Deserialize the engine
		if (core != -1) {
			runtime->setDLACore(core);
		}
		ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
		if (!engine)
		{
			std::cerr << "Error deserializing engine: " << enginePath << std::endl;
			return -1;
		}

		// Add the engine to the vector
		//engines.emplace_back(engine);
		engines.emplace(instanceName, engine);

		// Add the execution context to the vector
		//contexts.emplace_back(engine->createExecutionContext());
		contexts.emplace(instanceName, engine->createExecutionContext());
		if (!contexts[instanceName])
		{
			std::cerr << "Error creating execution context." << std::endl;
			return -1;
		}
		//bindings[instanceName] = std::map<std::string, std::unique_ptr<void *>>();
		//bindings.emplace(instanceName, std::map<std::string, std::unique_ptr<void *>>());

		for (const auto & input : instance["inputs"])
		{
			const auto & bindingName = input["name"].as<std::string>();
			size_t volume = 1;
			for (const auto & dim : input["shape"])
				volume *= dim.as<int>();

			void *bindingBuffer;
			cudaMallocManaged(&bindingBuffer, volume * sizeof(float));
			contexts[instanceName]->setTensorAddress(bindingName.c_str(), bindingBuffer);
			//bindings[instanceName].emplace(bindingName, std::unique_ptr<void, decltype(cuda_deleter)>(bindingBuffer, cuda_deleter));
			bindings[instanceName][bindingName] = bindingBuffer;
		}

		for (const auto & output : instance["outputs"])
		{
			const auto & bindingName = output["name"].as<std::string>();
			size_t volume = 1;
			for (const auto & dim : output["shape"])
				volume *= dim.as<int>();
			volume *= sizeof(float);

			void *bindingBuffer;
			cudaMallocManaged(&bindingBuffer, volume * sizeof(float));
			contexts[instanceName]->setTensorAddress(bindingName.c_str(), bindingBuffer);
			//bindings[instanceName].emplace(bindingName, std::unique_ptr<void, decltype(cuda_deleter)>(bindingBuffer, cuda_deleter));
			bindings[instanceName][bindingName] = bindingBuffer;
		}
		const auto & streamID = instance["streamID"].as<int>();

		if (!instanceGroups.count(streamID))
		{
			instanceGroups.emplace(streamID, std::make_tuple(nullptr, nullptr, nullptr, std::vector<std::string>()));
			cudaStreamCreate(&std::get<0>(instanceGroups[streamID]));
		}
		std::get<3>(instanceGroups[streamID]).emplace_back(instanceName);

		// Avoid capturing initialization calls by executing the enqueue function at least
		// once before starting CUDA graph capture
		contexts[instanceName]->enqueueV3(std::get<0>(instanceGroups[streamID]));
		cudaStreamSynchronize(std::get<0>(instanceGroups[streamID]));
		cudaEvent_t tempStartEvent, tempStopEvent;
		cudaEventCreate(&tempStartEvent);
		cudaEventCreate(&tempStopEvent);
		cudaEventRecord(tempStartEvent, std::get<0>(instanceGroups[streamID]));
		contexts[instanceName]->enqueueV3(std::get<0>(instanceGroups[streamID]));
		cudaEventRecord(tempStopEvent, std::get<0>(instanceGroups[streamID]));
		cudaStreamSynchronize(std::get<0>(instanceGroups[streamID]));
		float elapsedTime = 0.0f;
		cudaEventElapsedTime(&elapsedTime, tempStartEvent, tempStopEvent);
		std::cout << "Individual launch of " << instanceName << " without CUDA Graph took " << elapsedTime << " milliseconds" << std::endl << std::endl;
	}

	// Create CUDA events for timing
	cudaEvent_t startEvent, stopEvent;
	cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync);
	cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync);

	float totalFileReadTime = 0.0f;
	float totalInputCopyTime = 0.0f;
	float totalLaunchTime = 0.0f;
	float totalPrepareDLATime = 0.0f;
	float totalTimeWithoutFile = 0.0f;
	float totalTime = 0.0f;
	
	std::map<std::string, std::vector<float>> curInputData;

	size_t elementCount = 1024 * 1024 * 3;
	size_t image_size = 1024 * 1024;
	float* buffer = (float*)malloc(elementCount * sizeof(float));

	// Common mean and stddev used in mmcv
	float* mean = new float[3]{123.675, 116.28, 103.53};
	float* stddev = new float[3]{58.395, 57.12, 57.375};
	cudaStream_t stream_main = std::get<0>(instanceGroups[0]);

	// init 2 dla loadables
	auto dla_ctx = new CudlaContext(seg_dla_path.c_str(), 0);
	auto depth_dla_ctx = new CudlaContext(depth_dla_path.c_str(), 1);

	void* networkInputTensors[4];
	void* outputTensor[1];
	void *input_buf_0 = nullptr;
	void *input_buf_1 = nullptr;
	void *input_buf_2 = nullptr;
	void *input_buf_3 = nullptr;
	void *output_buf;
	void *depth_output_buf;
	void *depth_out_int = nullptr;

	// stream
	cudaStream_t streamSeg;
	cudaStream_t streamDepth;
	checkCudaErrors(cudaStreamCreateWithFlags(&streamSeg, cudaStreamNonBlocking));
	checkCudaErrors(cudaStreamCreateWithFlags(&streamDepth, cudaStreamNonBlocking));

	const int num_data_threads = 4;
	cudaStream_t data_streams[num_data_threads];
	for (int i = 0; i < num_data_threads; ++i) {
		checkCudaErrors(cudaStreamCreateWithFlags(&data_streams[i], cudaStreamNonBlocking));
	}

	cudaEvent_t m_start, m_end;
	float m_ms{0.0f};
	checkCudaErrors(cudaEventCreateWithFlags(&m_start, cudaEventBlockingSync));
	checkCudaErrors(cudaEventCreateWithFlags(&m_end, cudaEventBlockingSync));

	float ms{0.0f};
	float d_ms{0.0f};
	cudaEvent_t start, end;
	cudaEvent_t d_start, d_end;

	checkCudaErrors(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
	checkCudaErrors(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));

	checkCudaErrors(cudaEventCreateWithFlags(&d_start, cudaEventBlockingSync));
	checkCudaErrors(cudaEventCreateWithFlags(&d_end, cudaEventBlockingSync));

	
	pthread_t data_threads[num_data_threads];
	conversion_thead_pack data_tp[num_data_threads];

	for (int i = 0; i < num_data_threads; i++) {
		data_tp[i].stream = data_streams[i];
		data_tp[i].scale = scales[i];
		data_tp[i].size = feat_sizes[i];
	}

	// Launch the CUDA graph for the specified number of iterations and measure latency using CUDA events
	for (int i = 0; i < numIterations; i++)
	{
		auto wallClockBegin = std::chrono::steady_clock::now();
		for (const auto & [inputName, dataDesc] : inputs)
		{
			const auto & filePaths = std::get<1>(dataDesc);
			if (std::get<0>(dataDesc) == "image")
			{
				std::vector<unsigned char> image;
				unsigned width, height;

				unsigned error = lodepng::decode(image, width, height, filePaths[i % filePaths.size()]);
				if (error)
				{
					std::cerr << "Decoding " << filePaths[i % filePaths.size()] << " failed" << std::endl;
					std::cerr << "decode error " << error << ": " << lodepng_error_text(error) << std::endl;
					return -1;
				}
				// preprocess images
				preprocess_image(image.data(), buffer, image_size, stream_main);
			}
			else
			{
				std::cerr << "Unsupported data type at " << inputName << std::endl;
			}
		}
		auto wallClockFileEnd = std::chrono::steady_clock::now();

		for (const auto & instance : config["instances"])
		{
			for (const auto & input : instance["inputs"])
			{
				float * dstPtr = reinterpret_cast<float *>(bindings[instance["name"].as<std::string>()][input["name"].as<std::string>()]);
				std::memcpy(dstPtr, buffer, elementCount * sizeof(float));
			}
		}
		auto wallClockInputCopyEnd = std::chrono::steady_clock::now();

		const int num_threads = 3;

    	pthread_t threads[num_threads];
		thead_pack tp[num_threads];
		int index = 0;

		

		
		std::cout<< "before running engine"<<std::endl;
		checkCudaErrors(cudaEventRecord(m_start, stream_main));
		for (int i = 0; i < num_threads; i++) {
			auto instanceGroup = instanceGroups[i];
			tp[i].contexts = &contexts;
			tp[i].stream = std::get<0>(instanceGroup);
			tp[i].ins_names = std::get<3>(instanceGroup);
			if (pthread_create(&threads[i], NULL, launch_kernel, (void *) &tp[i])) {
				fprintf(stderr, "Error creating threadn");
				return 1;
			}
		}

		for (int i = 0; i < num_threads; i++) {
			if(pthread_join(threads[i], NULL)) {
				fprintf(stderr, "Error joining threadn");
				return 2;
			}
    	}
		checkCudaErrors(cudaEventRecord(m_end, stream_main));
		checkCudaErrors(cudaEventSynchronize(m_end));
		checkCudaErrors(cudaEventElapsedTime(&m_ms, m_start, m_end));
		std::cout<< "finish running engine"<<std::endl;

		auto wallClockEndBack = std::chrono::steady_clock::now();

		// get output for gpu engine
		float * output1  = reinterpret_cast<float *>(bindings["mtmi-backbone"]["input.72"]);
		float * output2  = reinterpret_cast<float *>(bindings["mtmi-backbone"]["input.148"]);
		float * output3  = reinterpret_cast<float *>(bindings["mtmi-backbone"]["input.224"]);
		float * output4  = reinterpret_cast<float *>(bindings["mtmi-backbone"]["input.292"]);


		checkCudaErrors(cudaMallocManaged(&input_buf_0, dla_ctx->getInputTensorSizeWithIndex(0)));
		checkCudaErrors(cudaMallocManaged(&input_buf_1, dla_ctx->getInputTensorSizeWithIndex(1)));
		checkCudaErrors(cudaMallocManaged(&input_buf_2, dla_ctx->getInputTensorSizeWithIndex(2)));
		checkCudaErrors(cudaMallocManaged(&input_buf_3, dla_ctx->getInputTensorSizeWithIndex(3)));

		checkCudaErrors(cudaMallocManaged(&depth_out_int, 4 * depth_dla_ctx->getOutputTensorSizeWithIndex(0)));
		checkCudaErrors(cudaMallocManaged(&output_buf, dla_ctx->getOutputTensorSizeWithIndex(0)));
		checkCudaErrors(cudaMallocManaged(&depth_output_buf, depth_dla_ctx->getOutputTensorSizeWithIndex(0)));
		


		data_tp[0].input = output1;
		data_tp[1].input = output2;
		data_tp[2].input = output3;
		data_tp[3].input = output4;

		data_tp[0].output = input_buf_0;
		data_tp[1].output = input_buf_1;
		data_tp[2].output = input_buf_2;
		data_tp[3].output = input_buf_3;
		

		for (int i = 0; i < num_data_threads; i++) {
			if (pthread_create(&data_threads[i], NULL, launch_conversion, (void *) &data_tp[i])) {
				fprintf(stderr, "Error creating threadn");
				return 1;
			}
		}

		for (int i = 0; i < num_data_threads; i++) {
			if(pthread_join(data_threads[i], NULL)) {
				fprintf(stderr, "Error joining threadn");
				return 2;
			}
    	}
		auto wallClockBeginDLA = std::chrono::steady_clock::now();
		std::vector<void *> cudla_inputs{input_buf_0, input_buf_1, input_buf_2, input_buf_3};
        std::vector<void *> cudla_outputs{output_buf};

		// the order is reversed for depth dla loadable
		std::vector<void *> depth_inputs{input_buf_3, input_buf_2, input_buf_1, input_buf_0};
		std::vector<void *> depth_outputs{depth_output_buf};
        

		// Register the GPU buffers for cuDLA, and Memset the output buffers
		// dla_ctx->bufferPrep(networkInputTensors , outputTensor, streamToRun);
		dla_ctx->bufferPrep(cudla_inputs, cudla_outputs, streamSeg);
		depth_dla_ctx->bufferPrep(depth_inputs, depth_outputs, streamDepth);
		checkCudaErrors(cudaStreamSynchronize(streamSeg));
		checkCudaErrors(cudaStreamSynchronize(streamDepth));
		
		checkCudaErrors(cudaEventRecord(start, streamSeg));
		checkCudaErrors(cudaEventRecord(d_start, streamDepth));

		// Submit Inference task to DLA
		dla_ctx->submitDLATask(streamSeg);
		depth_dla_ctx->submitDLATask(streamDepth);


		// Uncomment following part to log inference time for dla loadables

		// checkCudaErrors(cudaEventRecord(end, streamSeg));
		// checkCudaErrors(cudaEventRecord(d_end, streamDepth));
		// checkCudaErrors(cudaEventSynchronize(end));
		// checkCudaErrors(cudaEventSynchronize(d_end));
		// checkCudaErrors(cudaEventElapsedTime(&ms, start, end));
		// checkCudaErrors(cudaEventElapsedTime(&d_ms, d_start, d_end));
		// std::cout << "Backbone Inference time: " << m_ms << " ms" << std::endl;
		// std::cout << "Seg Inference time: " << ms << " ms" << std::endl;
		// std::cout << "Depth Inference time: " << d_ms << " ms" << std::endl;

		
		
		
		


		


		// // Uncomment following parts to save results:
		// int b = 1;
		// int h = 1024;
		// int w = 1024;
		// int c = 19;

		// std::ofstream outFile("output_seg_fp.txt");
		// std::ofstream depthFile("output_depth_fp.txt");

		// int8_t* int8_data = (int8_t *)output_buf;
		// int8_t* depth_data = (int8_t *)depth_output_buf;

		// // the scale in last depth layer to convert int8 back to float
		// convert_int8_to_float(depth_data, (float *)depth_out_int, 1024*1024, 0.00735941017047, streamDepth);

		// float* depth_out = (float *)depth_out_int;

		// std::cout<<"before writing seg"<<std::endl;
		// for (int i = 0; i < b*h*w*c; ++i) {
		// 	outFile << static_cast<int>(int8_data[i]) << "\n";
		// }
		// std::cout<<"before writing depth"<<std::endl;
		// for (int i = 0; i < b*h*w; ++i) {
		// 	depthFile << depth_out[i] << "\n";
		// }

		// checkCudaErrors(cudaStreamSynchronize(streamSeg));
		// checkCudaErrors(cudaStreamSynchronize(streamDepth));

		cudaEventRecord(stopEvent);

		auto wallClockEnd = std::chrono::steady_clock::now();

		auto wallClockDiff = std::chrono::duration_cast<std::chrono::microseconds>(wallClockFileEnd - wallClockBegin).count() / 1000.;
		totalFileReadTime += wallClockDiff;
		std::cout << "File read " << i + 1 << " took " << wallClockDiff << " milliseconds" << std::endl;

		wallClockDiff = std::chrono::duration_cast<std::chrono::microseconds>(wallClockBeginDLA - wallClockEndBack).count() / 1000.;
		totalPrepareDLATime += wallClockDiff;
		std::cout << "preparing DLA " << i + 1 << " took " << wallClockDiff << " milliseconds" << std::endl;

		wallClockDiff = std::chrono::duration_cast<std::chrono::microseconds>(wallClockInputCopyEnd - wallClockFileEnd).count() / 1000.;
		totalInputCopyTime += wallClockDiff;
		std::cout << "Input copy " << i + 1 << " took " << wallClockDiff << " milliseconds" << std::endl;

		float elapsedTime = 0.0f;
		cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
		totalLaunchTime += elapsedTime;
		std::cout << "Launch " << i + 1 << " took " << elapsedTime << " milliseconds" << std::endl;

		wallClockDiff = std::chrono::duration_cast<std::chrono::microseconds>(wallClockEnd - wallClockFileEnd).count() / 1000.;
		totalTimeWithoutFile += wallClockDiff;
		std::cout << "Wall clock time without file read for " << i + 1 << " took " << wallClockDiff << " milliseconds" << std::endl;

		wallClockDiff = std::chrono::duration_cast<std::chrono::microseconds>(wallClockEnd - wallClockBegin).count() / 1000.;
		totalTime += wallClockDiff;
		std::cout << "Wall clock time for " << i + 1 << " took " << wallClockDiff << " milliseconds" << std::endl;

		std::cout << std::endl;
	}

	std::cout << "Average file read time: " << totalFileReadTime / float(numIterations) << " milliseconds" << std::endl;
	std::cout << "Average input copy time: " << totalInputCopyTime / float(numIterations) << " milliseconds" << std::endl;
	std::cout << "Average launch time: " << totalLaunchTime / float(numIterations) << " milliseconds" << std::endl;
	std::cout << "Average wall clock time without file read: " << totalTimeWithoutFile / float(numIterations) << " milliseconds" << std::endl;
	std::cout << "Average wall clock time: " << totalTime / float(numIterations) << " milliseconds" << std::endl;
	std::cout << "Average prepare DLA clock time: " << totalPrepareDLATime / float(numIterations) << " milliseconds" << std::endl;

	// Clean up remaining resources
	//cudaGraphExecDestroy(graphExec);
	//cudaGraphDestroy(graph);
	//cudaStreamDestroy(stream);
	for (auto & [streamID, instanceGroup] : instanceGroups)
	{
		cudaGraphExecDestroy(std::get<2>(instanceGroup));
		cudaGraphDestroy(std::get<1>(instanceGroup));
		cudaStreamDestroy(std::get<0>(instanceGroup));
	}
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(end));

	checkCudaErrors(cudaEventDestroy(d_start));
	checkCudaErrors(cudaEventDestroy(d_end));

	cudaEventDestroy(m_start);
	cudaEventDestroy(m_end);

	for (auto & [instantName, bindingPair] : bindings)
	{
		for (auto & [bindingName, bindingPtr] : bindingPair)
		{
			cudaFree(bindingPtr);
		}
	}

	return 0;
}
