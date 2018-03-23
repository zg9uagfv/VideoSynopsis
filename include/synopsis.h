#ifndef VIDEO_SYNOPSIS_H
#define VIDEO_SYNOPSIS_H

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <list>
#include <stdio.h>
#include <algorithm>
#include <map>
#include <tracker/ImageTracker.h>
#include <tracker/PAOT.h>
#include <detector/Detector.h>
#include <detector/ssd_detect.hpp>

using namespace std;
using namespace cv;

/**
*保存团块的基本数据单元
*/
struct tube {
    //functions
    tube(Rect rect, int t, cv::Mat img); //构造函数
    ~tube(); //析构函数
    //variables
    Rect position; //团块在源图中位置
    int t_sec; //所在帧时间
    cv::Mat target;
};

class VideoSynopsisDB{
    public:
        void buildDB(vector<Tracking>& trackings, Mat& image, Mat& bgImg, int sec);
        void mergeDB(Mat& bgImg, vector<Mat>& output);
        void freeDB();
    private:
        map<int, list<tube *> > tubes_;
};

class VideoSynopsis {
    public:
        VideoSynopsis(string& input_file, string& output_file)
        {
            input_file_ = input_file;
            output_file_ = output_file;
            string model_file = "./models/ssd/VGGNet/VOC0712Plus/SSD_512x512_ft/deploy.prototxt";
            string weights_file = "./models/ssd/VGGNet/VOC0712Plus/SSD_512x512_ft/VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.caffemodel";
            string mean_file = "";
            string mean_value = "104,117,123";
            float confidence_threshold = 0.3;
            detector_ = std::make_shared<SSD_Detector>(model_file, weights_file, \
                                                mean_file, mean_value, confidence_threshold);
            ImageTracker tracker(detector_, std::make_shared<PAOT>());
            tracker_ = tracker;
        }
        ~VideoSynopsis(){};
    public:
        void process();
    private:
        void bgModeling();
    private:
        std::shared_ptr<Detector> detector_;
        ImageTracker tracker_;
        cv::Mat bgImg_;
        string input_file_;
        string output_file_;
        VideoSynopsisDB db_;
};
#endif //VIDEO_SYNOPSIS_H
