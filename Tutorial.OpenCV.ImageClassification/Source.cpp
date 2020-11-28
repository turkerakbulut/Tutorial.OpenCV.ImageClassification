#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
#include <fstream>
#include <iostream>
#include <cstdlib>

using namespace cv;
using namespace cv::dnn;
using namespace std;

/* Find best class for the blob (i. e. class with maximal probability) */
static void getMaxClass(const Mat &probBlob, int *classId, double *classProb)
{
	Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
	Point classNumber;
	minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
	*classId = classNumber.x;
}

//Reading class names from file
static std::vector<String> readClassNames(const char *filename)
{
	std::vector<String> classNames;
	std::ifstream fp(filename);
	if (!fp.is_open())
	{
		std::cerr << "File with classes labels not found: " << filename << std::endl;
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name.substr(name.find(' ') + 1));
	}
	fp.close();
	return classNames;
}

int main(int argc, char **argv)
{
	String config = "C:/Work/opencv-4.4.0/bvlc_googlenet.prototxt";
	String model = "C:/Work/opencv-4.4.0/bvlc_googlenet.caffemodel";
	String input = "C:/Work/opencv-4.4.0/2003_12_31_11_00_003b4c3b6d000d727d491f15cc69a883dd.jpg";
	String classNameFile = "C:/Work/opencv-4.4.0/classification_classes_ILSVRC2012.txt";

	std::vector<String> classNames = readClassNames(classNameFile.c_str());

	Net net = dnn::readNet(model, config);
	Mat image;
	VideoCapture capture;
	capture.open(0);
	if (capture.isOpened())
	{
		cout << "Capture is opened" << endl;
		for (;;)
		{
			capture >> image;

			Mat inputBlob = blobFromImage(image, 1.0f, Size(224, 224), Scalar(104, 117, 123), false); 
			net.setInput(inputBlob, "data");        
			Mat prob = net.forward("prob");        

			cv::TickMeter tick;
			for (int i = 0; i < 5; i++)
			{
				net.setInput(inputBlob, "data");       
				tick.start();
				prob = net.forward("prob");                     
				tick.stop();
			}
			int classId;
			double classProb;
			getMaxClass(prob, &classId, &classProb);
			
			std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
			std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
			std::cout << "Time: " << (double)tick.getTimeMilli() / tick.getCounter() << " ms (average from " << tick.getCounter() << " iterations)" << std::endl;
			imshow("Tutorial.OpenCV.ImageClassification", image);
			if (waitKey(10) >= 0)
				break;
		}
	}
	
	return 0;
} //main


