#include "synopsis.h"
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

void run(string &input_path, string &output_path);

void run(string &input_path, string &output_path)
{
    VideoSynopsis vs(input_path, output_path);
    vs.process();
    return;
}

int main(int argc, char* argv[])
{
    string input_file(argv[1]);
    string output_file(argv[2]);
    run(input_file, output_file);
    return 0;
}
