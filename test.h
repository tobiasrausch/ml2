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

    // Show image
    uint32_t index = 4769;
    std::cout << "Test image " << labels[index] << std::endl;
    cv::imwrite(c.outfile.string().c_str(), images[index]);


    
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    std::cout << std::endl;

    std::string image_path = cv::samples::findFile("starry_night.jpg");
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if(img.empty())
      {
	std::cout << "Could not read the image: " << image_path << std::endl;
	return 1;
      }
    cv::imshow("Display window", img);
    int k = cv::waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
      {
	cv::imwrite("starry_night.png", img);
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



