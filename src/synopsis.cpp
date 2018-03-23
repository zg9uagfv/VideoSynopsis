//
// Created by xin on 18-1-19.
//
#include "synopsis.h"
#include <opencv2/photo.hpp>
#include <opencv2/video/background_segm.hpp>

tube::tube(Rect rect, int t, cv::Mat img) :position(rect), t_sec(t)
{
    target = img.clone();
}

tube::~tube()
{
}

void VideoSynopsis::bgModeling()
{
    int frame_no = 0x01;
    cv::VideoCapture cap(input_file_);
    if (!cap.isOpened())
    {
        std::cout << "Read video Failed !" << std::endl;
        return;
    }
    cout<<"Background Modeling..."<<endl;
    Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2();
    Mat img, fgimg, fgmask;
    while (cap.read(img)) {
        if (frame_no >= 500/*frame_num_used*/) {
            break;
        }
        frame_no++;
        bg_model->apply(img, fgmask);
        fgimg = Scalar::all(0);
        img.copyTo(fgimg, fgmask);
        bg_model->getBackgroundImage(bgImg_);
    }
    cout<<"Background Model has been achieved!"<<endl;
    return;
}

void VideoSynopsisDB::buildDB(vector<Tracking>& trackings, Mat& image, Mat& bgImg, int sec)
{
    char stime[16];
    int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
    int m = 0, s = 0; //分、秒
    double fontScale = 0.5;
    int thickness = 1;
    int baseline = 0;
    m = sec / 60;
    s = sec % 60;
    sprintf(stime, "%02d:%02d", m, s);
    Size textSize = getTextSize(stime, fontFace, fontScale, thickness, &baseline);
    cv::Scalar color(0, 255, 255);
    for (auto trackingIt = trackings.begin(); trackingIt != trackings.end(); ++trackingIt)
    {
        int width = 0x00, height = 0x00;
        if ((trackingIt->bb.x1() < 0) || (trackingIt->bb.y1() < 0)) {
            continue;
        }

        if ((trackingIt->bb.width < 0) || (isnan(trackingIt->bb.width))) {
            continue;
        }

        if ((trackingIt->bb.height < 0) || (isnan(trackingIt->bb.height))) {
            continue;
        }

        if ((trackingIt->bb.x1()+trackingIt->bb.width) > image.cols) {
            width = image.cols - trackingIt->bb.x1();
        }else{
            width = trackingIt->bb.width;
        }

        if ((trackingIt->bb.y1()+trackingIt->bb.height) > image.rows) {
            height = image.rows - trackingIt->bb.y1();
        }else{
            height = trackingIt->bb.height;
        }

        Rect rect(trackingIt->bb.x1(), trackingIt->bb.y1(), width, height);
        Mat frame = image(rect);
        cv::Point pt(0, textSize.height);
        cv::putText(frame, stime, pt, CV_FONT_HERSHEY_DUPLEX, fontScale, color, thickness, 8);
        map<int, list<tube *> >::iterator iter = tubes_.find(trackingIt->ID);
        if (iter != tubes_.end()) {
            (iter->second).push_back(new tube(rect, sec, frame));
        }else{
            list<tube *> list;
            list.push_back(new tube(rect, sec, frame));
            tubes_.insert(make_pair(trackingIt->ID, list));
        }
    }
    return;
}

void VideoSynopsisDB::mergeDB(Mat& bgImg, vector<Mat>& output)
{
    cout<<"Merging  Database!"<<endl;
    tube *tmp = NULL; //临时tube指针
    Mat frame;
    while (!tubes_.empty()) {
        frame = bgImg.clone();
        map<int, list<tube *> >::iterator iter = tubes_.begin();
        while(iter != tubes_.end()) {
            if (0x00 == (iter->second).size()) {
                tubes_.erase(iter->first);
            }else{
                tmp = *((iter->second).begin()); //begin() is a iterator
                Mat src = tmp->target;
                Mat src_mask = 255 * Mat::ones(src.rows, src.cols, src.depth());
                Point center((int)(tmp->position.x+(tmp->position.width)/2), \
                            (int)(tmp->position.y+(tmp->position.height)/2));
                Mat mixed_clone;
                seamlessClone(src, frame, src_mask, center, mixed_clone, cv::MIXED_CLONE);
                frame = mixed_clone.clone();
                delete tmp; //释放tmp指向的tube的图像内存
                (iter->second).pop_front();
            }
            iter++;
        }
        output.push_back(frame);
    }
    cout<<"Merge Database has been achieved!"<<endl;
    return;
}


void VideoSynopsisDB::freeDB()
{
    map<int, list<tube *> >::iterator iter = tubes_.begin();
    while(iter != tubes_.end()) {
        for (list<tube *>::iterator iter2 = (iter->second).begin(); iter2 != (iter->second).end();) {
            delete *iter2;
        }
    }
    return;
}

void VideoSynopsis::process()
{
    this->bgModeling();
    cv::VideoCapture cap(input_file_);
    if (!cap.isOpened())
    {
        std::cout << "Read video Failed !" << std::endl;
        return;
    }
    Mat img;
    int frame_rate = cap.get(CV_CAP_PROP_FPS);
    int frame_no = 0x01;
    while (cap.read(img))
    {
        vector<Tracking> trackings = tracker_.detectAndTrack(img);
        db_.buildDB(trackings, img, bgImg_, frame_no/frame_rate);
        frame_no++;
    }

    vector<Mat> result;
    db_.mergeDB(bgImg_, result);

    VideoWriter writer;
    Size size = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
                     (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    writer.open(output_file_, 0, frame_rate/1.5, size, true);
    vector<Mat>::iterator iter = result.begin();
    while (iter != result.end()) {
        writer.write(*iter);
        iter++;
    }
    writer.release();
    return;
}
