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


  static std::vector<int64_t> k_size = {2, 2};
  static std::vector<int64_t> p_size = {0, 0};
  static c10::optional<int64_t> divisor_override;
  
  class LeNet5Impl : public torch::nn::Module {
  public:
    LeNet5Impl() {
      conv_ = torch::nn::Sequential(
      torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 6, 5)),
      torch::nn::Functional(torch::tanh),
      torch::nn::Functional(torch::avg_pool2d,
                            /*kernel_size*/ torch::IntArrayRef(k_size),
                            /*stride*/ torch::IntArrayRef(k_size),
                            /*padding*/ torch::IntArrayRef(p_size),
                            /*ceil_mode*/ false,
                            /*count_include_pad*/ false,
                            divisor_override),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 16, 5)),
      torch::nn::Functional(torch::tanh),
      torch::nn::Functional(torch::avg_pool2d,
                            /*kernel_size*/ torch::IntArrayRef(k_size),
                            /*stride*/ torch::IntArrayRef(k_size),
                            /*padding*/ torch::IntArrayRef(p_size),
                            /*ceil_mode*/ false,
                            /*count_include_pad*/ false,
                            divisor_override),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 120, 5)),
      torch::nn::Functional(torch::tanh));
      register_module("conv", conv_);

      full_ = torch::nn::Sequential(
      torch::nn::Linear(torch::nn::LinearOptions(120, 84)),
      torch::nn::Functional(torch::tanh),
      torch::nn::Linear(torch::nn::LinearOptions(84, 10)));
      register_module("full", full_);
    }
  
    torch::Tensor forward(at::Tensor x) {
      auto output = conv_->forward(x);
      output = output.view({x.size(0), -1});
      output = full_->forward(output);
      output = torch::log_softmax(output, -1);
      return output;
    }
    
  private:
    torch::nn::Sequential conv_;
    torch::nn::Sequential full_;
  };
  TORCH_MODULE(LeNet5);
  
  
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
	  cv::resize(images[i], images[i], cv::Size(32, 32));  // Resize to 32x32 size
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

    // model
    LeNet5Impl model;
    model.to(device);

    // optimizer
    double learning_rate = 0.01;
    double weight_decay = 0.0001;
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(learning_rate).weight_decay(weight_decay).momentum(0.5));

    // Training
    int epochs = 100;
    for (int epoch = 0; epoch < epochs; ++epoch) {
      model.train();

      // Batches
      int idx = 0;
      for (auto& batch : (*train_loader)) {
	optimizer.zero_grad();
	torch::Tensor prediction = model.forward(batch.data);

	// test data
	// std::cout << prediction << std::endl;
	// std::cout << batch.target << std::endl;

	torch::Tensor loss = torch::nll_loss(prediction, batch.target.squeeze(1));
	loss.backward();
	optimizer.step();

	if (idx % 10 == 0) {
	  std::cout << "Epoch: " << epoch << ", Batch: " << idx << ", Loss: " << loss.item<float>() << std::endl;
	}
	++idx;
      }

      // Test data set for evaluation
      model.eval();
      unsigned long total_correct = 0;
      float avg_loss = 0.0;
      for (auto& batch : (*test_loader)) {
	torch::Tensor prediction = model.forward(batch.data);
	torch::Tensor loss = torch::nll_loss(prediction, batch.target.squeeze(1));
	avg_loss += loss.sum().item<float>();
	auto pred = std::get<1>(prediction.detach_().max(1));
	total_correct += static_cast<unsigned long>(pred.eq(batch.target.view_as(pred)).sum().item<long>());
      }
      avg_loss /= test_data_set.size().value();
      double accuracy = (static_cast<double>(total_correct) / test_data_set.size().value());
      std::cout << "Test Avg. Loss: " << avg_loss << ", Accuracy: " << accuracy << std::endl;
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



