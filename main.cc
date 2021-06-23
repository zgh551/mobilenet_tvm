/*
 * mobilenet module
 * */
// tvm 
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
// opencv 
#include <opencv4/opencv2/opencv.hpp>
// system
#include <cstdio>
#include <fstream>
#include <sys/time.h>

double GetCurTime(void)
{
    struct timeval tm;
    gettimeofday(&tm, 0);
    return tm.tv_usec + tm.tv_sec * 1000000;
}


int main(int argc, char *argv[])
{
    if (argc > 1)
    {
        LOG(INFO) << "[mobilenet tvm]:Image Path: " << argv[1];
        LOG(INFO) << "[mobilenet tvm]:Dynamic Lib Path: " << argv[2];
        LOG(INFO) << "[mobilenet tvm]:Parameter Path: " << argv[3];
    }
    else
    {
        LOG(INFO) << "executor [img] [mod lib] [mod param]";
        return -1;
    }
    LOG(INFO) << "[mobilenet tvm]:Soft Version: V" << SOFTWARE_VERSION;

    // read the image
    cv::Mat image;
    uint32_t img_c, img_h, img_w;

    image = cv::imread(argv[1]);
    if(image.data == nullptr){
        LOG(INFO) << "[mobilenet tvm]:Image don't exist!";
        return 0;
    }
    else{
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        image.convertTo(image, CV_32FC3, 2.0/255, -1);
        cv::resize(image, image, cv::Size(224, 224), 0, 0, cv::INTER_CUBIC);

        img_c = image.channels();
        img_h = image.rows;
        img_w = image.cols;
        LOG(INFO) << "[mobilenet tvm]:---Load Image--";
        LOG(INFO) << "[mobilenet tvm]:Image size: " << img_h << " X " << img_w;
        // cv::imshow("mobilenet image", gray_image);
        // cv::waitKey(0);
    }

    // create tensor
    DLTensor *x;
    DLTensor *y;
    int input_ndim  = 4;
    int output_ndim = 2;
    int64_t input_shape[4]  = {1, img_c, img_h, img_w};
    int64_t output_shape[2] = {1, 1000};

    int dtype_code  = kDLFloat;
    int dtype_bits  = 32;
    int dtype_lanes = 1;
    int device_id   = 0;
#ifdef CPU
    int device_type = kDLCPU;
    LOG(INFO) << "[mobilenet tvm]:---Device Type Configure:CPU---";
#elif OPENCL
    int device_type = kDLOpenCL;
    LOG(INFO) << "[mobilenet tvm]:---Device Type Configure:OPENCL---";
#endif

    TVMByteArray params_arr;
    DLDevice dev{static_cast<DLDeviceType>(device_type), device_id};

    // allocate the array space
    TVMArrayAlloc(input_shape, input_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
    TVMArrayAlloc(output_shape, output_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);

    // the memory space allocate
    //std::vector<float> x_input(gray_image.rows * gray_image.cols);
    //std::vector<float> y_output(10);
        
    // load the mobilenet dynamic lib
    LOG(INFO) << "[mobilenet tvm]:---Load Dynamic Lib---";
    tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile(argv[2]);
    // get the mobilenet module
    tvm::runtime::Module mod = mod_dylib.GetFunction("mobilenet")(dev);

    // load the mobilenet module parameters
    LOG(INFO) << "[mobilenet tvm]:---Load Parameters---";
    std::ifstream params_in(argv[3], std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();
    // get load parameters function
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);

    LOG(INFO) << "[mobilenet tvm]:---Set Input---";
    // get set input data function
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    // copy image data to cpu memory space
    //memcpy(x_input.data(), gray_image.data, gray_image.rows * gray_image.cols * sizeof(float));
    // from cpu memory space copy data to gpu memory space
    TVMArrayCopyFromBytes(x, image.data, img_c * img_h * img_w * sizeof(float));
    set_input("Input3", x);

    LOG(INFO) << "[mobilenet tvm]:---Run---";
    // get run function
    tvm::runtime::PackedFunc run = mod.GetFunction("run");
    double t1 = GetCurTime();
    run();
    double t2 = GetCurTime();
    LOG(INFO) << "[mobilenet tvm]:---Executor Time:" << t2 - t1 << "[us]";

    LOG(INFO) << "[mobilenet tvm]:---Get Output---";
    // get output data function
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
    get_output(0, y);
    //TVMArrayCopyToBytes(y, y_output.data(), 10 * sizeof(float));

    //auto result = static_cast<float *>(y->data);
    /*for (int i = 0; i < 10; i++)
    {
        LOG(INFO) << y_output[i];
    }*/

    TVMArrayFree(x);
    TVMArrayFree(y);

    return 0;
}
