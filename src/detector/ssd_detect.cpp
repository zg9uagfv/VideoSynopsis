#include <detector/ssd_detect.hpp>
#include <util/Detection.h>

SSD_Detector::SSD_Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value,
                   float confidence_threshold) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    threshold_ = confidence_threshold;

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(weights_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    SetMean(mean_file, mean_value);
}

/**
*判断两矩形是否重叠
*/
bool SSD_Detector::isOverlap(const Rect & a, const Rect & b)
{
    const Rect * l_rect = &a,
            *r_rect = &b;
    if (a.x > b.x) {
        const Rect * tmp = l_rect;
        l_rect = r_rect;
        r_rect = tmp;
    }

    if (l_rect->width < r_rect->x - l_rect->x)
        return false;
    else if (l_rect->y <= r_rect->y && l_rect->height >= r_rect->y - l_rect->y)
        return true;
    else if (l_rect->y > r_rect->y && r_rect->height >= l_rect->y - r_rect->y)
        return true;
    else
        return false;
}


/**
*前景位置矩形块合并 - 减少单元数
*/
void SSD_Detector::mergeRects(list<Rect> & rects)
{
    int x = 0, y = 0, width = 0, height = 0;//临时变量
    for (list<Rect>::iterator i = rects.begin(); i != rects.end(); ) {
        bool merged = false;//多引入一个变量判断i是否被merge非常有用！
        list<Rect>::iterator j = i;
        for (j++; j != rects.end(); j++) {
            if (isOverlap(*i, *j)) {
                if (i->x < j->x) {
                    x = i->x;
                    width = max(j->x - i->x + j->width, i->width);
                }
                else {
                    x = j->x;
                    width = max(i->x - j->x + i->width, j->width);
                }

                if (i->y < j->y) {
                    y = i->y;
                    height = max(j->y - i->y + j->height, i->height);
                }
                else {
                    y = j->y;
                    height = max(i->y - j->y + i->height, j->height);
                }

                //合并
                j->x = x;
                j->y = y;
                j->width = width;
                j->height = height;

                i = rects.erase(i);//删除被合并项。注意：删除前者（i）更新后者（j）！
                merged = true;
            }
        }
        if (!merged)
            i++;
    }
}

std::vector<Detection> SSD_Detector::do_detect(const cv::Mat& image)
{
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                        input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(image, &input_channels);

    net_->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* result_blob = net_->output_blobs()[0];
    const float* result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    std::vector<Detection> detections;
    std::list<Rect> rects;
    for (int k = 0; k < num_det; ++k) {
        if (result[0] == -1 || result[2] < 0.1) {
            // Skip invalid detection.
            result += 7;
            continue;
        }
        vector<float> det(result, result + 7);
        int label = static_cast<int>(det[1]);
        /* bus || car || person */
        if (label == 7 || label == 6 || label == 15) {
            int x,y,x2,y2;
            x = static_cast<int>(det[3]*image.cols);
            y = static_cast<int>(det[4]*image.rows);
            if (x < 0.0) {
                x = 0;
            }
            if (y < 0.0) {
                y = 0;
            }
            x2 =  static_cast<int>(det[5]*image.cols);
            y2 =  static_cast<int>(det[6]*image.rows);
            if (x2 > image.cols) {
                x2 = image.cols;
            }
            if (y2 > image.rows) {
                y2 = image.rows;
            }
            rects.push_back(Rect(x, y, x2-x, y2-y));
        }
        result += 7;
    }
    mergeRects(rects);
    list<Rect>::iterator iter = rects.begin();
    for (; iter != rects.end(); iter++) {
        Detection detection(int(0), 0.5, BoundingBox((2*iter->x + iter->width)/2 ,
                                                        (2*iter->y+iter->height)/2,
                                                        (iter->width),
                                                        (iter->height)));
        detections.push_back(detection);
    }
    return detections;
}

/* Load the mean file in binaryproto format. */
void SSD_Detector::SetMean(const string& mean_file, const string& mean_value)
{
    cv::Scalar channel_mean;
    if (!mean_file.empty()) {
        CHECK(mean_value.empty()) <<
            "Cannot specify mean_file and mean_value at the same time";
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        CHECK_EQ(mean_blob.channels(), num_channels_)
            << "Number of channels of mean file doesn't match input layer.";

        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<cv::Mat> channels;
        float* data = mean_blob.mutable_cpu_data();
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }

        /* Merge the separate channels into a single image. */
        cv::Mat mean;
        cv::merge(channels, mean);

        /* Compute the global mean pixel value and create a mean image
            * filled with this value. */
        channel_mean = cv::mean(mean);
        mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
    }
    if (!mean_value.empty()) {
        CHECK(mean_file.empty()) <<
            "Cannot specify mean_file and mean_value at the same time";
        stringstream ss(mean_value);
        vector<float> values;
        string item;
        while (getline(ss, item, ',')) {
            float value = std::atof(item.c_str());
            values.push_back(value);
        }
        CHECK(values.size() == 1 || values.size() == num_channels_) <<
            "Specify either 1 mean_value or as many as channels: " << num_channels_;

        std::vector<cv::Mat> channels;
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
                cv::Scalar(values[i]));
            channels.push_back(channel);
        }
        cv::merge(channels, mean_);
    }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void SSD_Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void SSD_Detector::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}
