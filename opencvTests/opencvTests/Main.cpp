#include <opencv2\opencv.hpp>

using namespace cv;
/*
int main(){
	Mat image;
	Mat imageRect(300,300,CV_8UC3);

	image = imread("c:\\teste.png");

	printf("linhas : %d\nColunas :%d\n",imageRect.rows,imageRect.cols);

	printf("linhas : %d\nColunas :%d\n",image.rows,image.cols);

    Vec3b color;

	
	for(int i = 0 ; i < imageRect.rows;i++){
		for(int j = 0;j < imageRect.cols;j++){

			color = image.at<Vec3b>(Point(i+50,j+50));
			imageRect.at<Vec3b>(Point(i,j)) = color;

		}
	}
	
	
	for(;;){
		waitKey(5);
		imshow("Image",image);
		imshow("imageRect",imageRect);
	};
	return 0;
}*/

#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
 
int main(int argc, char *argv[])
{
    cv::Mat frame;
    cv::Mat fore;
    cv::VideoCapture cap(0);

	cap.set(CV_CAP_PROP_FRAME_HEIGHT,400);
	cap.set(CV_CAP_PROP_FRAME_WIDTH,320);

	cv::BackgroundSubtractorMOG2 bg;
    bg.nmixtures = 3;
    bg.bShadowDetection = false;

    std::vector<std::vector<cv::Point> > contours,contours1;
 
	vector<Vec4i> hierarchy;
    //Create a structuring element
    int erosion_size = 2;
    Mat element = getStructuringElement(cv::MORPH_CROSS,
                                        cv::Size(erosion_size +1 ,erosion_size + 1),
                                        cv::Point(erosion_size, erosion_size) );
    for(;;)
    {
		cap.read(frame);
        bg.operator ()(frame,fore);
		cv::erode(fore,fore,element);
		cv::dilate(fore,fore,element);

		cv::findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_TC89_KCOS);
	    cv::drawContours(frame,contours,-1,cv::Scalar(0,0,255),1.3);
		cv::imshow("frame 1",frame);

		waitKey(30);
	}
    return 0;
}