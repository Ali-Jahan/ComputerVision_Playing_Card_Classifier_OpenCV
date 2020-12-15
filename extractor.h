// extractor.h - Declarations for the Extractor class
// Author: Yang Hu
#pragma once
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;
// Extractor - given an isolated image of a card (i.e., the image height and width is 
// approximately the same as the height and width of the card itself), extracts two 
// subimages representing the rank and the suit of the card. The subimages are taken
// from the top left corner of the card.
class Extractor {
private:
    Mat src_gray, src;
    RNG rng;
    string rank_window = "rank";
    string suit_window = "suit";
    string source_window = "source";
    string contour_window = "contour";
    bool enableShowResult;
    float rankBrYPosition = -1;
    float rankBrXPosition = -1;
    // the contour representing suit should be within the range of error
    // of the contour representing rank 
    int error = 5;
    // findSubImages
    // Precondition: src is set to a correctly formatted image of a playing card.
    //
    // Postcondition: performs Canny edge detection, find all contours in the image,
    // then apprximate contours to polygons. Then, filter out possible contours representing
    // the rank based on contour dimensions, position, and area. Similarly, find the contour
    // representing the suit. Finally, extract sub-images from the contours and set
    // the rank and suit fileds of the object appropriately. If both rank and suit and found,
    // and showResult is set to true, display contours and subiumages. 
    void findSubImages();
    // showResult
    // Precondition: suit, rank, and drawing fields of the object are initialized
    // Postcondition: display suit, rank, and drawing in named windows
    void showResult();

public:
    // drawing displays the contours identified by the Extractor on the original image
    Mat rank, suit, drawing;
    // constructor
    // Precondiditon: none
    // Postcondition: initializes an RNG instance
    Extractor(bool showResult);
    // extract
    // Precondition: input is a correctly formatted representation of a single playing
    // card, where the border of the image is approximately the border of the playing card.
    //
    // Postcondition: Crop the input image and only keep approximately the top-left 1/4 
    // of the image and set src to the resultant image.
    // If appropriate contours representing the rank and suit of the card
    // are found in src, sets the rank field of the object to the rank sub-image extracted
    // from the card, the suit field of the object to the suit sub-image of the card, and 
    // display the contours and subimages on seperate windows if showResult is set to true.
    // Otherwise, return false; the rank and suit fields of the object should not be used.
    bool extract(Mat input);
};