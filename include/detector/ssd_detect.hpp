#ifndef _SSD_DETECTOR_H_
#define _SSD_DETECTOR_H_

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <boost/thread/thread.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#ifdef _MSC_VER
#include <boost/config/compiler/visualc.hpp>
#endif
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>
#include <stdlib.h>
#include "detector/Detector.h"

using namespace std;
using namespace cv;
using namespace caffe;  // NOLINT(build/namespaces)

class SSD_Detector:public Detector{
    public:
        SSD_Detector(const string& model_file,
               const string& weights_file,
               const string& mean_file,
               const string& mean_value,
               float confidence_threshold);

        std::vector<Detection> do_detect(const cv::Mat& img);

    private:
        void SetMean(const string& mean_file, const string& mean_value);
        void WrapInputLayer(std::vector<cv::Mat>* input_channels);
        void Preprocess(const cv::Mat& img,
                      std::vector<cv::Mat>* input_channels);
        bool isOverlap(const Rect & a, const Rect & b);
        void mergeRects(list<Rect> & rects);
    private:
        boost::shared_ptr<Net<float> > net_;
        cv::Size input_geometry_;
        int num_channels_;
        cv::Mat mean_;
        float threshold_;

};

#endif
