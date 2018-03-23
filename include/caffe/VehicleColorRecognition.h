#ifndef VEHICLECOLORRECOGNITION_H
#define VEHICLECOLORRECOGNITION_H

/************************************************************************/
/* 输入（cv::Mat）： 车辆颜色图片;
   输出（std::string）： 车辆名*/
/************************************************************************/

#include <string>
#include <opencv2/highgui/highgui.hpp>

std::string recognizeVehicleColor(const cv::Mat& bgrImg);

#endif // VEHICLECOLORRECOGNITION_H