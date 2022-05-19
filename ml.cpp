#define _SECURE_SCL 0
#define _SCL_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>

#define BOOST_DISABLE_ASSERTS

#ifdef PROFILE
#include "gperftools/profiler.h"
#endif

#include "opencv2/opencv.hpp"

#include "version.h"
#include "mnist.h"

using namespace ml;

inline void
displayUsage() {
  std::cout << "Usage: ml <command> <arguments>" << std::endl;
  std::cout << std::endl;
  std::cout << "    mnist          MNIST example" << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
}

int main(int argc, char **argv) {
    if (argc < 2) { 
      printTitle("Ml");
      displayUsage();
      return 0;
    }

    if ((std::string(argv[1]) == "version") || (std::string(argv[1]) == "--version") || (std::string(argv[1]) == "--version-only") || (std::string(argv[1]) == "-v")) {
      std::cout << "Ml version: v" << mlVersionNumber << std::endl;
      std::cout << " using Boost: v" << BOOST_VERSION / 100000 << "." << BOOST_VERSION / 100 % 1000 << "." << BOOST_VERSION % 100 << std::endl;
      std::cout << " using HTSlib: v" << hts_version() << std::endl;
      std::cout << " using OpenCV: v" << CV_VERSION << std::endl;
      std::cout << " using PyTorch: v" << TORCH_VERSION << std::endl;
      
      return 0;
    }
    else if ((std::string(argv[1]) == "help") || (std::string(argv[1]) == "--help") || (std::string(argv[1]) == "-h") || (std::string(argv[1]) == "-?")) {
      printTitle("Ml");
      displayUsage();
      return 0;
    }
    else if ((std::string(argv[1]) == "warranty") || (std::string(argv[1]) == "--warranty") || (std::string(argv[1]) == "-w")) {
      displayWarranty();
      return 0;
    }
    else if ((std::string(argv[1]) == "license") || (std::string(argv[1]) == "--license") || (std::string(argv[1]) == "-l")) {
      bsd();
      return 0;
    }
    else if ((std::string(argv[1]) == "mnist")) {
      return mnist(argc-1,argv+1);
    }
    std::cerr << "Unrecognized command " << std::string(argv[1]) << std::endl;
    return 1;
}
