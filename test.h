#ifndef MATCHES_H
#define MATCHES_H


#include <iostream>
#include <fstream>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/timer/timer.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/icl/split_interval_map.hpp>
#include <boost/filesystem.hpp>

#include <htslib/faidx.h>
#include <htslib/vcf.h>
#include <htslib/sam.h>
#include <htslib/tbx.h>

#include <torch/torch.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>


namespace ml {

  // Config arguments
  struct TestConfig {
    boost::filesystem::path outfile;
    boost::filesystem::path path;
  };

  struct Net: torch::nn::Module {
    Net() {
      conv1_1 = register_module("conv1_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 3).padding(1)));
      conv1_2 = register_module("conv1_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 3).padding(1)));
      conv2_1 = register_module("conv2_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(20, 30, 3).padding(1)));
      conv2_2 = register_module("conv2_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(30, 40, 3).padding(1)));
      conv3_1 = register_module("conv3_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(40, 50, 3).padding(1)));
      conv3_2 = register_module("conv3_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(50, 60, 3).padding(1)));
      conv3_3 = register_module("conv3_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(60, 70, 3).padding(1)));
      conv4_1 = register_module("conv4_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(70, 80, 3).padding(1)));
      conv4_2 = register_module("conv4_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(80, 90, 3).padding(1)));
      conv4_3 = register_module("conv4_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(90, 100, 3).padding(1)));
      conv5_1 = register_module("conv5_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(100, 110, 3).padding(1)));
      conv5_2 = register_module("conv5_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(110, 120, 3).padding(1)));
      conv5_3 = register_module("conv5_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(120, 130, 3).padding(1)));
      fc1 = register_module("fc1", torch::nn::Linear(130, 50));
      fc2 = register_module("fc2", torch::nn::Linear(50, 20));
      fc3 = register_module("fc3", torch::nn::Linear(20, 10));
    }
    
    torch::Tensor forward(torch::Tensor x) {
      x = torch::relu(conv1_1->forward(x));
      x = torch::relu(conv1_2->forward(x));
      x = torch::max_pool2d(x, 2);
      
      x = torch::relu(conv2_1->forward(x));
      x = torch::relu(conv2_2->forward(x));
      x = torch::max_pool2d(x, 2);
      
      x = torch::relu(conv3_1->forward(x));
      x = torch::relu(conv3_2->forward(x));
      x = torch::relu(conv3_3->forward(x));
      x = torch::max_pool2d(x, 2);
      
      x = torch::relu(conv4_1->forward(x));
      x = torch::relu(conv4_2->forward(x));
      x = torch::relu(conv4_3->forward(x));
      x = torch::max_pool2d(x, 2);
      
      x = torch::relu(conv5_1->forward(x));
      x = torch::relu(conv5_2->forward(x));
      x = torch::relu(conv5_3->forward(x));
      
      x = x.view({-1, 130});
      
      x = torch::relu(fc1->forward(x));
      x = torch::relu(fc2->forward(x));
      x = fc3->forward(x);
      
      return torch::log_softmax(x, 1);
    }

    torch::nn::Conv2d conv1_1{nullptr};
    torch::nn::Conv2d conv1_2{nullptr};
    torch::nn::Conv2d conv2_1{nullptr};
    torch::nn::Conv2d conv2_2{nullptr};
    torch::nn::Conv2d conv3_1{nullptr};
    torch::nn::Conv2d conv3_2{nullptr};
    torch::nn::Conv2d conv3_3{nullptr};
    torch::nn::Conv2d conv4_1{nullptr};
    torch::nn::Conv2d conv4_2{nullptr};
    torch::nn::Conv2d conv4_3{nullptr};
    torch::nn::Conv2d conv5_1{nullptr};
    torch::nn::Conv2d conv5_2{nullptr};
    torch::nn::Conv2d conv5_3{nullptr};

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
  };

  torch::Tensor CvImageToTensor(const cv::Mat& image, torch::DeviceType device) {
    assert(image.channels() == 1);
    std::vector<int64_t> dims{static_cast<int64_t>(1), static_cast<int64_t>(image.rows), static_cast<int64_t>(image.cols)};

    torch::Tensor tensor_image = torch::from_blob(image.data, torch::IntArrayRef(dims), torch::TensorOptions().dtype(torch::kFloat).requires_grad(false)).clone();
    return tensor_image.to(device);
  }

  class CustomDataset : public torch::data::Dataset<CustomDataset>
  {
  private:
    std::vector<torch::Tensor> images, labels;

    public:
        explicit CustomDataset(std::vector<cv::Mat> const& img, std::vector<unsigned char> const& lb, torch::DeviceType const& device) {
	  images.resize(lb.size());
	  labels.resize(lb.size());
	  for(uint32_t i = 0; i < lb.size(); ++i) {
	    images[i] = CvImageToTensor(img[i], device);
	    labels[i] = torch::tensor(static_cast<int64_t>(lb[i]), torch::TensorOptions().dtype(torch::kLong).device(device));
	  }
	};

    torch::data::Example<> get(size_t index) {
      return {images[index], labels[index]};
    }

    torch::optional<size_t> size() const {
      return labels.size();
    }
  };
  
  template <class T>
  bool read_header(T* out, std::istream& stream) {
    auto size = static_cast<std::streamsize>(sizeof(T));
    T value;
    if (!stream.read(reinterpret_cast<char*>(&value), size)) {
      return false;
    } else {
      // flip endianness
      *out = (value << 24) | ((value << 8) & 0x00FF0000) |
	((value >> 8) & 0X0000FF00) | (value >> 24);
      return true;
    }
  }

  inline void
  _readLabels(std::string const& labels_file_name, std::vector<unsigned char>& labels) {
    std::ifstream labels_file(labels_file_name, std::ios::binary | std::ios::binary);
    labels_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    if (labels_file) {
      uint32_t magic_num = 0;
      uint32_t num_items = 0;
      if (read_header(&magic_num, labels_file) && read_header(&num_items, labels_file)) {
	labels.resize(static_cast<size_t>(num_items));
	labels_file.read(reinterpret_cast<char*>(labels.data()), num_items);
      }
    }
  }

  inline void
  _readImages(std::string const& images_file_name, std::vector<cv::Mat>& images) {
    std::ifstream image_file(images_file_name, std::ios::binary | std::ios::binary);
    image_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    if (image_file) {
      uint32_t magic_num = 0;
      uint32_t num_items = 0;
      uint32_t rows_ = 0;
      uint32_t columns_ = 0;
      if (read_header(&magic_num, image_file) && read_header(&num_items, image_file) && read_header(&rows_, image_file) && read_header(&columns_, image_file)) {
	images.resize(num_items);
	cv::Mat img(static_cast<int>(rows_), static_cast<int>(columns_), CV_8UC1);

	for (uint32_t i = 0; i < num_items; ++i) {
	  image_file.read(reinterpret_cast<char*>(img.data), static_cast<std::streamsize>(img.size().area()));
	  img.convertTo(images[i], CV_32F);
	  images[i] /= 255;  // normalize
	  cv::resize(images[i], images[i], cv::Size(16, 16));  // Resize to 32x32 size
	}
      }
    }
  }

  
  template<typename TConfigStruct>
  inline int testRun(TConfigStruct& c) {
#ifdef PROFILE
    ProfilerStart("test.prof");
#endif

    boost::filesystem::path train_images = c.path / "train-images-idx3-ubyte";
    boost::filesystem::path train_labels = c.path / "train-labels-idx1-ubyte";
    boost::filesystem::path test_images = c.path / "t10k-images-idx3-ubyte";
    boost::filesystem::path test_labels = c.path / "t10k-labels-idx1-ubyte";
    torch::DeviceType device = torch::cuda::is_available() ? torch::DeviceType::CUDA : torch::DeviceType::CPU;

    // Load train images
    std::vector<unsigned char> labels;
    std::vector<cv::Mat> images;
    _readLabels(train_labels.string(), labels);
    _readImages(train_images.string(), images);
    auto train_data_set = CustomDataset(images, labels, device);
    auto train_loader = torch::data::make_data_loader(train_data_set.map(torch::data::transforms::Stack<>()), torch::data::DataLoaderOptions().batch_size(256).workers(8));

    // Show example image
    cv::imshow(std::to_string(labels[300]), images[300]);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // Load test images
    std::vector<unsigned char> tlabels;
    std::vector<cv::Mat> timages;
    _readLabels(test_labels.string(), tlabels);
    _readImages(test_images.string(), timages);
    auto test_data_set = CustomDataset(timages, tlabels, device);
    auto test_loader = torch::data::make_data_loader(test_data_set.map(torch::data::transforms::Stack<>()), torch::data::DataLoaderOptions().batch_size(1024).workers(8));

    // Show example image
    cv::imshow(std::to_string(tlabels[300]), timages[300]);
    cv::waitKey(0);
    cv::destroyAllWindows();

    auto net = std::make_shared<Net>();
    torch::optim::SGD optimizer(net->parameters(), 0.01);
    for(uint32_t epoch=0; epoch<10; ++epoch) {
      net->train(); // Training mode
      uint32_t bidx = 0;
      for (auto& batch: *train_loader) {
	optimizer.zero_grad();
	torch::Tensor prediction = net->forward(batch.data);
	torch::Tensor loss = torch::nll_loss(prediction, batch.target);
	loss.backward();
	optimizer.step();
	if (bidx % 10 == 0) {
	  std::cout << "Epoch: " << epoch << ", Batch: " << bidx << ", Loss: " << loss.item<float>() << std::endl;
	}
	++bidx;
      }

      net->eval();  // switch to the training mode
      float avg_loss = 0.0;
      for (auto& batch : *test_loader) {
	torch::Tensor prediction = net->forward(batch.data);
	torch::Tensor loss = torch::nll_loss(prediction, batch.target);
	avg_loss += loss.sum().item<float>();
      }
      avg_loss /= test_data_set.size().value();
      std::cout << "Test Avg. Loss: " << avg_loss << std::endl;
    }
    

#ifdef PROFILE
    ProfilerStop();
#endif
  
    // End
    boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();
    std::cout << '[' << boost::posix_time::to_simple_string(now) << "] Done." << std::endl;;
    return 0;
  }


  int test(int argc, char **argv) {
    TestConfig c;
    
    // Define generic options
    boost::program_options::options_description generic("Generic options");
    generic.add_options()
      ("help,?", "show help message")
      ("outfile", boost::program_options::value<boost::filesystem::path>(&c.outfile)->default_value("out.png"), "outfile")
      ;
    
    // Define hidden options
    boost::program_options::options_description hidden("Hidden options");
    hidden.add_options()
      ("input-file", boost::program_options::value<boost::filesystem::path>(&c.path), "MNIST path")
      ;
    
    boost::program_options::positional_options_description pos_args;
    pos_args.add("input-file", -1);
    
    // Set the visibility
    boost::program_options::options_description cmdline_options;
    cmdline_options.add(generic).add(hidden);
    boost::program_options::options_description visible_options;
    visible_options.add(generic);
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(cmdline_options).positional(pos_args).run(), vm);
    boost::program_options::notify(vm);
    

    // Check command line arguments
    if ((vm.count("help")) || (!vm.count("input-file"))) { 
      std::cout << std::endl;
      std::cout << "Usage: ml " << argv[0] << " [OPTIONS] <test>" << std::endl;
      std::cout << visible_options << "\n";
      return 0;
    }

    // Show cmd
    boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();
    std::cout << '[' << boost::posix_time::to_simple_string(now) << "] ";
    std::cout << "ml ";
    for(int i=0; i<argc; ++i) { std::cout << argv[i] << ' '; }
    std::cout << std::endl;
    
    return testRun(c);
  }

}

#endif



