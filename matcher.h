// Matcher class
// performs matching between two images, in order
// to quantify the difference between two images,
// and using this value, it decides which of the 
// reference images are the closest match.
// Author: Ali Jahangirnezhad

#pragma once
#include "extractor.h"
#include <limits>

class Matcher {
private:
	// resize Width to resize all ranks and suits for matching
	const int RESIZE_WIDTH = 20;
	// resize Height to resize all ranks and suits for matching
	const int RESIZE_HEIGHT = 30;
	// threshold for converting to binary
	const int BW_THRESHOLD = 128;
	// an error threshhold for Rank - if the error exceeds this amount
	// then the classifier has failed to classify the image
	double MAX_ERROR_RANK;
	// an error threshhold for Suit - if the error exceeds this amount
	// then the classifier has failed to classify the image
	double MAX_ERROR_SUIT;
	// rank images previously extracted, as reference.
	// read in the same order as NAMES
	Mat ranks[13];
	// suit images previously extracted, as reference.
	// read in the same order as SUIT_names
	Mat suits[4];
	// names of each card in string format
	const String RANK_NAMES[13] = { "Ten", "Ace", "Two", "Three",
					   "Four", "Five", "Six", "Seven",
					   "Eight", "Nine", "Jack",
					   "Queen", "King" };
	// different suits in a deck of cards
	const String SUIT_NAMES[4] = { "Hearts", "Diamonds", "Spades", "Clubs" };
	// reads the previously extracted reference images,
	// and stores them in ranks and suits
	void readTrainFiles();
	// a function to return a value which is a measure of difference of two images
	// Parameters:  test - The image to be tested against
	//				input - The input to be tested
	// Postconditions:	Returns a value indicating how different the images are.
	//					0 = no difference
	double matchValue(const Mat& test, const Mat& input);

public:
	// default constructor
	// calls readTrainFiles() on construction
	// Parameters: maxErrRank - Maximum allowed error to classify rank as match
	//						defaults to 12
	//			   maxErrSuit - Maximum allowed error to classify suit as match
	//						defaults to 9
	Matcher(double maxErrRank = 2000, double maxErrSuit = 1000);
	// a function to return the best matching rank or suit
	// Parameters:	input - The input image to find best match for
	//				isRank - A boolean showing if the image is a rank image (suit if false)
	// Postconditions: Returns the name of the rank or suit that is closes to the input image
	String getBestMatch(const Mat& input, bool isRank);
};

