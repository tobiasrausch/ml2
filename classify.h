#ifndef CLASSIFY_H
#define CLASSIFY_H


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
  struct ClassifyConfig {
    boost::filesystem::path train_path;
    boost::filesystem::path test_path;
  };

  class SarsDataset : public torch::data::Dataset<SarsDataset>
  {
  private:
    std::vector<torch::Tensor> images, labels;

    public:
        explicit SarsDataset(std::vector<cv::Mat> const& img, std::vector<uint8_t> const& lb, torch::DeviceType const& device) {
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


  struct Net23 : torch::nn::Module {
    Net23()
      : conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
        conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
        fc1(320, 50),
        fc2(50, 10) {
      register_module("conv1", conv1);
      register_module("conv2", conv2);
      register_module("conv2_drop", conv2_drop);
      register_module("fc1", fc1);
      register_module("fc2", fc2);
    }
    
    torch::Tensor forward(torch::Tensor x) {
      x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
      x = torch::relu(
		      torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
      x = x.view({-1, 320});
      x = torch::relu(fc1->forward(x));
      x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
      x = fc2->forward(x);
      return torch::log_softmax(x, /*dim=*/1);
    }
    
    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Dropout2d conv2_drop;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
  };


  template<typename TValue>
  inline bool
  _loadImageLabel(boost::filesystem::path& path, std::vector<cv::Mat>& images, std::vector<TValue>& labels, std::map<std::string, TValue>& lbmap) {
    TValue lbidx = 0;
    if ( !boost::filesystem::exists(path)) return false;
    boost::filesystem::directory_iterator end_itr; 
    for(boost::filesystem::directory_iterator itr(path); itr != end_itr; ++itr) {
      if (!is_directory(itr->status())) {
	if (itr->path().extension().string() == ".png") {
	  std::string fn = itr->path().filename().string();
	  std::string classname = fn.substr(0, fn.find('_'));
	  if (lbmap.find(classname) == lbmap.end()) {
	    lbmap.insert(std::make_pair(classname, lbidx));
	    ++lbidx;
	  }
	  TValue lb = lbmap[classname];
	  labels.push_back(lb);
	  cv::Mat imgIn = cv::imread(itr->path().string().c_str(), cv::IMREAD_GRAYSCALE);
	  if (imgIn.empty()) {
	    std::cerr << "Failed imread(): " << itr->path().string() << std::endl;
	    return false;
	  }
	  
	  // Debug
	  //std::cout << itr->path().string() << std::endl;
	  //std::cout << "Width : " << imgIn.cols << std::endl;
	  //std::cout << "Height: " << imgIn.rows << std::endl;
	  //cv::resize(imgIn, imgIn, cv::Size(28, 28)); // resize
	  //cv::imwrite("test.png", imgIn);

	  // Convert and normalize
	  cv::Mat img;
	  imgIn.convertTo(img, CV_32F);
          img /= 255;  // normalize
          cv::resize(img, img, cv::Size(28, 28)); // resize

	  // Debug images
	  //for(int j=0; j<img.rows; ++j) {
	  //for (int i=0; i<img.cols; ++i) std::cout << (float) img.at<float>(j,i) << ",";
	  //std::cout << std::endl;
	  //}
	  images.push_back(img);
	}
      }
    }
    std::cout << "Parsed " << (int32_t) lbidx << " new classes." << std::endl;
    return true;
  }
  
  template<typename TConfigStruct>
  inline int classifyRun(TConfigStruct& c) {
#ifdef PROFILE
    ProfilerStart("test.prof");
#endif

    // Device
    torch::DeviceType device = torch::cuda::is_available() ? torch::DeviceType::CUDA : torch::DeviceType::CPU;
    
    // Labels and images
    typedef uint8_t TLabelClasses;
    std::vector<TLabelClasses> labels;
    std::vector<cv::Mat> images;

    // Load images and labels
    std::map<std::string, TLabelClasses> lbmap;
    if (!_loadImageLabel(c.train_path, images, labels, lbmap)) return 1;
    std::cout << "Parsed " << images.size() << " train images." << std::endl;
    std::cout << "Parsed " << labels.size() << " train labels." << std::endl;
    //std::cout << "Image properties: " <<  images[0].shape  << std::endl;
    
    // Show example image
    //cv::imshow(std::to_string(labels[0]), images[0]);
    //cv::waitKey(0);
    //cv::destroyAllWindows();
    
    // Load train images
    auto train_data_set = SarsDataset(images, labels, device);
    auto train_loader = torch::data::make_data_loader(train_data_set.map(torch::data::transforms::Stack<>()), torch::data::DataLoaderOptions().batch_size(64).workers(8));

    // Load test images
    std::vector<TLabelClasses> tlabels;
    std::vector<cv::Mat> timages;
    if (!_loadImageLabel(c.test_path, timages, tlabels, lbmap)) return 1;
    std::cout << "Parsed " << timages.size() << " test images." << std::endl;
    std::cout << "Parsed " << tlabels.size() << " test labels." << std::endl;

    // Show example image
    //cv::imshow(std::to_string(tlabels[0]), timages[0]);
    //cv::waitKey(0);
    //cv::destroyAllWindows();

    auto test_data_set = CustomDataset(timages, tlabels, device);
    auto test_loader = torch::data::make_data_loader(test_data_set.map(torch::data::transforms::Stack<>()), torch::data::DataLoaderOptions().batch_size(64).workers(8));

    auto net = std::make_shared<Net23>();
    torch::optim::SGD optimizer(net->parameters(), 0.01);
    for(uint32_t epoch=0; epoch<100; ++epoch) {
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

      net->eval();  // switch to the testing mode
      float avg_loss = 0.0;
      int32_t correct = 0;
      for (auto& batch : *test_loader) {
	torch::Tensor prediction = net->forward(batch.data);
	torch::Tensor loss = torch::nll_loss(prediction, batch.target);
	avg_loss += loss.sum().item<float>();
	auto targets = batch.target.to(device);
	correct += prediction.argmax(1).eq(targets).sum().template item<int64_t>();
      }
      avg_loss /= test_data_set.size().value();
      double acc = (double) correct / (double) test_data_set.size().value();
      std::cout << "Test Avg. Loss: " << avg_loss << ", Accuracy: " << acc << std::endl;
    }
    

#ifdef PROFILE
    ProfilerStop();
#endif

    // End
    boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();
    std::cout << '[' << boost::posix_time::to_simple_string(now) << "] Done." << std::endl;;
    return 0;
  }


  int classify(int argc, char **argv) {
    ClassifyConfig c;
    
    // Define generic options
    boost::program_options::options_description generic("Generic options");
    generic.add_options()
      ("help,?", "show help message")
      ("test,t", boost::program_options::value<boost::filesystem::path>(&c.test_path), "path with testing images")
      ;
    
    // Define hidden options
    boost::program_options::options_description hidden("Hidden options");
    hidden.add_options()
      ("input-file", boost::program_options::value<boost::filesystem::path>(&c.train_path), "path with training images")
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
      std::cout << "Usage: ml " << argv[0] << " [OPTIONS] ./images/" << std::endl;
      std::cout << visible_options << "\n";
      return 0;
    }

    // Show cmd
    boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();
    std::cout << '[' << boost::posix_time::to_simple_string(now) << "] ";
    std::cout << "ml ";
    for(int i=0; i<argc; ++i) { std::cout << argv[i] << ' '; }
    std::cout << std::endl;
    
    return classifyRun(c);
  }

}

#endif



