#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>

#include <iostream>


using namespace cv;
using namespace std;


void detectAndDisplay(Mat& frame);
Mat skin_retouching(Mat& frame);


String cascadeName = "./haarcascade_frontalface_alt.xml";  
CascadeClassifier cascade;

// String nestedCascadeName = "./haarcascade_eye_tree_eyeglasses.xml"; 

int main(int, char**)
{
	double scale = 1.3;

	if (!cascade.load(cascadeName))// Проверка на наличие каскада
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
	}

	Mat frame;
	//--- INITIALIZE VIDEOCAPTURE
	VideoCapture cap;
	// open the default camera using default API
	// cap.open(0);
	// OR advance usage: select any API backend
	int deviceID = 0;             // 0 = open default camera
	int apiID = cv::CAP_ANY;      // 0 = autodetect default API
	// open selected camera using selected API
	cap.open(deviceID, apiID);
	// check if we succeeded
	if (!cap.isOpened()) {
		cerr << "ERROR! Unable to open camera\n";
		return -1;
	}
	//--- GRAB AND WRITE LOOP
	cout << "Start grabbing" << endl
		<< "Press any key to terminate" << endl;
	for (;;)
	{
		// wait for a new frame from camera and store it into 'frame'
		cap.read(frame);

		// check if we succeeded
		if (!frame.empty())
		{
			// Модуль сглаживания и ретуши кожи
			skin_retouching(frame);

			// show live and wait for a key with timeout long enough to show images
			detectAndDisplay(frame);
			if (waitKey(5) >= 0)
			break;
		}
	}

	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}


Mat skin_retouching(Mat& frame) {

	Mat dst;

	int value1 = 1, value2 = 2;

	int dx = value1 * 5;    
	double fc = value1 * 12.5;  
	int p = 50;  
	Mat temp1, temp2, temp3, temp4;

	// Filter  
	bilateralFilter(frame, temp1, dx, fc, fc);

	temp2 = (temp1 - frame + 128);
  
	GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);

	temp4 = frame + 2 * temp3 - 255;

	dst = (frame * (100 - p) + temp4 * p) / 100;
	dst.copyTo(frame);
	
	return frame;
}


void detectAndDisplay(Mat& frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	std::vector<Rect> faces;
	cascade.detectMultiScale(frame_gray, faces);
	for (size_t i = 0; i < faces.size(); i++) {
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(0, 0, 255), 1);
	}
	//-- Show what you got
	imshow("Capture - Face detection", frame);
}
