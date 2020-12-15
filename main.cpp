// The main driver of the program.
// It uses the system's default camera (ID = 1),
// and needs a standard deck of playing cards.
// The program performs all the actions automatically,
// and running the program shows your camera feed, 
// along with the contours used, rank and suit image,
// and finally produces the rank and suit of the 
// card in the image. 
// Authors: Yang Hu - Ali Jahangirnezhad - Sonia Saravanan

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <opencv2/videoio.hpp>
#include "extractor.h"
#include "matcher.h"
#include <time.h>

using namespace cv;
using namespace std;

// minimum contour area to classify as card
double MIN_CONTOUR_AREA = 3000;

// a function to apply preprocessing to each frame
// Parameters:	image - Mat object of one frame of the camera
//				foreGround - temporary image to hold thresholded and contoured image of frame
//				contours - the contours extracted from the image
//				hierarchy - the contor hirearchy
// PostCondition: the images passed into this function is converted to grayscale,
//					stored in foreGround, blurred, converted to black and white,
//					and then its edges are detected via canny edge detector.
//					As a final step, the contours of the image are extracted
//					and saved in "contours", and the hierarchy is saved in "hierarchy"
void preprocessFrame(Mat& image, Mat& foreGround, vector<vector<Point>>& contours, vector<Vec4i>& hierarchy) {
	// convert to grayscale
	cvtColor(image, foreGround, COLOR_BGR2GRAY);
	// blurr
	GaussianBlur(foreGround, foreGround, Size(5, 5), 0);
	// to black and white
	threshold(foreGround, foreGround, 128, 255, THRESH_BINARY);
	// Canny ED
	Canny(foreGround, foreGround, THRESH_OTSU, THRESH_OTSU);
	// find contours
	findContours(foreGround, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);
}

int main() {
	// matching object initalization
	Matcher matcher(12, 9);
	// image holding each frame
	Mat frame;
	// initialize video capture
	VideoCapture cap;
	// default camera id = 0
	int deviceID = 0;
	// autodetect default API
	int apiID = cv::CAP_ANY;
	// open selected camera using selected API
	cap.open(deviceID, apiID);
	// check if we succeeded in opening the camera
	if (!cap.isOpened()) {
		cerr << "Cannot access camera!\n";
		return -1;
	}
	// rank and suit extractor
	Extractor ext(false);
	// images to hold each frame
	Mat image, foreGround;
	// contours vector
	vector<vector<Point> > contours;
	// largest contours list
	// a vector, in order to keep algorithm open to future expansion 
	// so that it detects several cards in the same frame.
	vector<vector<Point> > largestContours;
	// contour hierarchy
	vector<Vec4i> hierarchy;
	for (;;)
	{
		// wait for a new frame from camera and store it into 'frame'
		cap.read(frame);
		// check if we succeeded
		if (frame.empty()) {
			cerr << "Blank Frame\n";
			break;
		}
		// save frame data in image
		image = frame.clone();
		// preprocess this frame
		// refer to the documentation for preprocess
		preprocessFrame(image, foreGround, contours, hierarchy);
		// find the largest contours
		double area_max = 0;
		for (int i = 0; i < contours.size(); i++) {  // get the largest contour
			double area = fabs(contourArea(contours[i]));
			if (area >= area_max) {
				area_max = area;
				largestContours.clear();
				largestContours.push_back(contours[i]);
			}
		}

		// a mask to hold area inside the contour
		Mat mask = Mat::zeros(image.rows, image.cols, CV_8UC1);
		// if the max contour area is bigger than a preset amount
		// to prevent unwanted small contours

		if (area_max > MIN_CONTOUR_AREA) {
			// draw contours
			drawContours(image, largestContours, 0, Scalar(255, 0, 0), 2);
			drawContours(mask, largestContours, 0, Scalar(255, 0, 0), FILLED);
		}
		// a vector of vector of points to hold the contours
		vector<vector<Point> > contours_poly(1);
		if (largestContours.size() != 0 && area_max > MIN_CONTOUR_AREA) {
			approxPolyDP(largestContours.at(0), contours_poly[0], 0.2, true);
			Rect bounds = boundingRect(contours_poly[0]);
			Mat temp = image(bounds);
			if (ext.extract(temp)) {
				namedWindow("Rank", WINDOW_NORMAL);
				imshow("Rank", ext.rank);
				namedWindow("Suit", WINDOW_NORMAL);
				imshow("Suit", ext.suit);
			}
			namedWindow("Card Extracted", WINDOW_NORMAL);
			// matcher
			String rank = matcher.getBestMatch(ext.rank, true);
			String suit = matcher.getBestMatch(ext.suit, false);
			String output_text = "Rank : " + rank + " ---- Suit : " + suit;
			// output image
			Mat output = image(bounds).clone();
			// print the rank and suit on the output image
			putText(output, output_text, Point(70, 20), 2, 0.5, Scalar(0, 255, 255), 2, 8);
			imshow("Card Extracted", output);
			cout << "Rank: " << rank << "---- Suit: " << suit << endl;
		}
		// show live and wait for a key with timeout long enough to show images
		imshow("Live", image);
		if (waitKey(5) >= 0) {
			break;
		}
	}
	return 0;
}