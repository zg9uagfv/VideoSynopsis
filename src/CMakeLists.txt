link_directories("../lib")
link_directories("/usr/local/lib")
link_directories("/usr/lib")
link_directories("/usr/local/cuda-8.0/lib64")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall -O0")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -O0")
add_executable(video_synopsis
    ./util/Tracking.cpp 
    ./util/BoundingBox.cpp
    ./util/classification.cpp
    ./util/Detection.cpp
    ./synopsis.cpp
    ./tracker/predictor/particle/ParticleFilter.cpp
    ./tracker/predictor/particle/Particle.cpp
    ./tracker/predictor/particle/ParticlePredictor.cpp
    ./tracker/predictor/Predictor.cpp
    ./tracker/predictor/StationaryPredictor.cpp
    ./tracker/predictor/kalman/KalmanPredictor.cpp
    ./tracker/Affinity.cpp
    ./tracker/RandomTracker.cpp
    ./tracker/ImageTracker.cpp
    ./tracker/PAOT.cpp
    ./tracker/Tracker.cpp
    ./detector/Detector.cpp
    ./detector/ssd_detect.cpp
    main.cpp
    )
find_package(OpenCV REQUIRED)
include_directories(../include /usr/local/cuda-8.0/include/)
target_link_libraries(video_synopsis ${OpenCV_LIBS} glog gflags protobuf
    boost_system boost_filesystem boost_regex caffe lapack)
