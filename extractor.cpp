// extractor.cpp - implementations of the Extractor class.
// Author: Yang Hu
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "extractor.h"

using namespace cv;
using namespace std;

// constructor
// Precondiditon: none
// Postcondition: initializes an RNG instance
Extractor::Extractor(bool enableShowResult) {
    this->enableShowResult = enableShowResult;
    RNG rng(12345);
}

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
bool Extractor::extract(Mat input) {
    src = Mat(input, Rect(0, 0, input.cols / 4, input.rows / 3.5));
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    blur(src_gray, src_gray, Size(3, 3));
    findSubImages();
    return !(rank.empty() || suit.empty());
}

// showResult
// Precondition: suit, rank, and drawing fields of the object are initialized
// Postcondition: display suit, rank, and drawing in named windows
void Extractor::showResult() {
    namedWindow(source_window);
    imshow(source_window, src);
    namedWindow(contour_window);
    imshow(contour_window, drawing);
    namedWindow(rank_window);
    imshow(rank_window, rank);
    namedWindow(suit_window);
    imshow(suit_window, suit);
    waitKey();
}
// findSubImages
// Precondition: src is set to a correctly formatted image of a playing card.
//
// Postcondition: performs Canny edge detection, find all contours in the image,
// then apprximate contours to polygons. Then, filter out possible contours representing
// the rank based on contour dimensions, position, and area. Similarly, find the contour
// representing the suit. Finally, extract sub-images from the contours and set
// the rank and suit fileds of the object appropriately. If both rank and suit and found,
// and showResult is set to true, display contours and subiumages. 
void Extractor::findSubImages()
{
    Mat canny_output;
    // adaptive threshold
    Canny(src_gray, canny_output, THRESH_OTSU, THRESH_OTSU);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(canny_output, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);

    vector<vector<Point> > contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());

    for (size_t i = 0; i < contours.size(); i++)
    {
        approxPolyDP(contours[i], contours_poly[i], 3, true);
        boundRect[i] = boundingRect(contours_poly[i]);
    }

    drawing = Mat::zeros(canny_output.size(), CV_8UC3);
    Mat extractMask = Mat::zeros(src.size(), CV_8UC3);

    int area = src.cols * src.rows;

    // find the contour representing rank
    for (size_t i = 0; i < contours.size(); i++)
    {

        int height = -boundRect[i].tl().y + boundRect[i].br().y;
        int width = -boundRect[i].tl().x + boundRect[i].br().x;

        // filter out the contour representing rank and suit by checking the dimensions
        // ratio, area, and the position of the contour
        if (height * width < area / 10 || height / width > 4 || height / width  < 1
            || boundRect[i].br().x >(3 * src.cols / 4)
            || boundRect[i].br().y > 3 * src.rows / 4) {
            continue;
        }
        else {
            // update rank br position
            rankBrYPosition = boundRect[i].br().y;
            rankBrXPosition = boundRect[i].br().x;
            // create img from contour
            drawContours(extractMask, contours_poly, i, Scalar(255), 0);

            Mat imageROI = src.clone();
            src.copyTo(imageROI, extractMask);
            rank = imageROI(boundRect[i]);

            // draw contour for window
            Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
            drawContours(drawing, contours_poly, (int)i, color);
            rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2);
            break;
        }
    }
    // return if no contour representing rank is found
    if (rankBrYPosition < 0) {
        cout << "did not find number";
        return;
    }
    // find suit
    for (size_t i = 0; i < contours.size(); i++) {
        int height = -boundRect[i].tl().y + boundRect[i].br().y;
        int width = -boundRect[i].tl().x + boundRect[i].br().x;

        // filter out the contour representing suit by checking the dimensions and the bottom right position
        // of the contour
        if (height * width < area / 25 || height / width > 4 || height / width < 1 || boundRect[i].br().y <= rankBrYPosition + error || boundRect[i].br().x >= rankBrXPosition + src.cols / 6) {
            continue;
        }
        else {

            // create img from contour
            drawContours(extractMask, contours_poly, i, Scalar(255), 0); // This is a OpenCV function

            Mat imageROI = src.clone();
            src.copyTo(imageROI, extractMask); // 'src' is the image you used to compute the contours.
            suit = imageROI(boundRect[i]);

            // draw contour for window
            Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
            drawContours(drawing, contours_poly, (int)i, color);
            rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2);
            break;
        }
    }
    // only show results if both rank and suit sub-images are found
    if (!rank.empty() && !suit.empty() && enableShowResult) {
        showResult();
    }
}