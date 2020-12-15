// matcher.cpp - implementations of the Extractor class.
// Author: Ali Jahangirnezhad

#include "matcher.h"
#include <iostream>

// reads the previously extracted reference images,
// and stores them in ranks and suits
void Matcher::readTrainFiles() {
	// read rank samples
	ranks[0] = imread("0.png");
	ranks[1] = imread("A.png");
	ranks[2] = imread("2.png");
	ranks[3] = imread("3.png");
	ranks[4] = imread("4.png");
	ranks[5] = imread("5.png");
	ranks[6] = imread("6.png");
	ranks[7] = imread("7.png");
	ranks[8] = imread("8.png");
	ranks[9] = imread("9.png");
	ranks[10] = imread("J.png");
	ranks[11] = imread("Q.png");
	ranks[12] = imread("K.png");
	// save the suits 
	suits[0] = imread("hearts.png");
	suits[1] = imread("diamonds.png");
	suits[2] = imread("spades.png");
	suits[3] = imread("clubs.png");
}

// default constructor
// calls readTrainFiles() on construction
// Parameters: maxErr - Maximum allowed error to classify card as match
//						defaults to 2000
Matcher::Matcher(double maxErrRank, double maxErrSuit)
{
	MAX_ERROR_RANK = maxErrRank;
	MAX_ERROR_SUIT = maxErrSuit;
	readTrainFiles();
}

// a function to return a value which is a measure of difference of two images
// Parameters:  test - The image to be tested against
//				input - The input to be tested
// Postconditions:	Returns a value indicating how different the images are.
//					0 = no difference
double Matcher::matchValue(const Mat& test, const Mat& input)
{
	Mat test_clone = test.clone();
	Mat input_clone = input.clone();
	// if input image is empty, just return -1
	if (input.rows < 1 || input.cols < 1) {
		return -1;
	}
	// share size to resize everything to
	Size s(RESIZE_WIDTH, RESIZE_HEIGHT);
	// resize so both have the same size, in order to calculate difference
	resize(input_clone, input_clone, s);
	resize(test_clone, test_clone, s);
	// convert to greyScale
	cvtColor(test_clone, test_clone, COLOR_BGR2GRAY);
	cvtColor(input_clone, input_clone, COLOR_BGR2GRAY);
	//// blurr images for easier distinguishing
	//GaussianBlur(input_clone, input_clone, Size(5, 5), 0);
	//GaussianBlur(test_clone, test_clone, Size(5, 5), 0);
	// threshold to convert to binary (black and white)
	threshold(test_clone, test_clone, BW_THRESHOLD, 255, THRESH_BINARY);
	threshold(input_clone, input_clone, BW_THRESHOLD, 255, THRESH_BINARY);
	// temporary matrix to hold the difference matrix
	Mat temp;
	// calculates the difference between two matrices, saves results in temp
	absdiff(test_clone, input_clone, temp);	
	// different pixels have the value 255, the rest are 0
	//double value = sum(temp)[0]/ 255;
	// norm value is divided by 1000 just to make the numbers smaller
	double value = norm(test_clone, input_clone) / 1000;
	return value;
}

// a function to return the best matching rank or suit
// Parameters:	input - The input image to find best match for
//				isRank - A boolean showing if the image is a rank image (suit if false)
// Postconditions: Returns the name of the rank or suit that is closes to the input image
String Matcher::getBestMatch(const Mat& input, bool isRank)
{
	// initialize minimum error to max double
	double minError = std::numeric_limits<double>::max();
	// index of the minimum error 
	int minIndex = -1;
	// error difference between two images
	double error = 0;
	if (isRank) {
		// find best rank match
		for (int i = 0; i < size(ranks); i++) {
			error = matchValue(ranks[i], input);
			if (error < minError) {
				minIndex = i;
				minError = error;
			}
		}
		std::cout << minError << std::endl;
		if (minError < MAX_ERROR_RANK) {
			return RANK_NAMES[minIndex];
		}
		else {
			return "(Rank not Classified)";
		}
	}
	else {
		// find best suit match
		for (int i = 0; i < size(suits); i++) {
			error = matchValue(suits[i], input);
			if (error < minError) {
				minIndex = i;
				minError = error;
			}
		}
		std::cout << minError << std::endl;
		if (minError < MAX_ERROR_SUIT) {
			return SUIT_NAMES[minIndex];
		}
		else {
			return "(Suit not Classified)";
		}
	}
}
