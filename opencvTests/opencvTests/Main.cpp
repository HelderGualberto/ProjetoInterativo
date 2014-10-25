#include <opencv2\opencv.hpp>
#include <process.h>
using namespace cv;
/*
Mat getImagePart(Mat cameraFeed,Rect rect){

	Mat imageRect(rect.height,rect.width,CV_8UC3);

	printf("linhas : %d\nColunas :%d\n",imageRect.rows,imageRect.cols);

	//printf("linhas : %d\nColunas :%d\n",image.rows,image.cols);

    Vec3b color;

	
	for(int i = 0 ; i < imageRect.rows;i++){
		for(int j = 0;j < imageRect.cols;j++){

			color = cameraFeed.at<Vec3b>(Point(j+rect.x,i+rect.y));
			imageRect.at<Vec3b>(Point(j,i)) = color;

		}
	}
	return imageRect;
}

int main(){
	Mat image,imageRect;
	Rect rect;
	VideoCapture cap(0);
	//352,288

	cap.set(CV_CAP_PROP_FRAME_HEIGHT,288);
	cap.set(CV_CAP_PROP_FRAME_WIDTH,352);
	rect.x = 0;
	rect.y = 0;
	rect.height = 100;
	rect.width = 100;
	cap.read(image);
	
	printf("h: %d\nw: %d\n",image.cols,image.rows);
	for(int i = 0;;i+=10){
		for(int i = 0; i < image.cols-rect.x;i++){
			waitKey(5);
			cap.read(image);
			imageRect = getImagePart(image,rect);
			imshow("Image",image);
			imshow("imageRect",imageRect);
			rect.x++;
		};
		rect.x = 0;
	}
	return 0;
}*/
/*
#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
 */


/*
int main(int argc, char *argv[])
{
    cv::Mat frame,lastFrame,frameDiff;
    
    cv::VideoCapture cap(0);

	cap.set(CV_CAP_PROP_FRAME_HEIGHT,400);
	cap.set(CV_CAP_PROP_FRAME_WIDTH,320);

    std::vector<std::vector<cv::Point> > contours,contours1;
	vector<Vec4i> hierarchy;
    //Create a structuring element
    int erosion_size = 2;
    Mat element = getStructuringElement(cv::MORPH_CROSS,
                                        cv::Size(erosion_size +1 ,erosion_size + 1),
                                        cv::Point(erosion_size, erosion_size) );
    for(;;)
    {
		cap.read(lastFrame);
		waitKey(10);
		cap.read(frame);


		absdiff(lastFrame,frame,frameDiff);
		
		
		cv::dilate(frameDiff,frameDiff,element);
		cv::erode(frameDiff,frameDiff,element);

		cvtColor(frameDiff,frameDiff,CV_RGB2GRAY,1);

		threshold(frameDiff,frameDiff,20, 255,CV_THRESH_BINARY);

		cv::imshow("frame 1",frameDiff);
		uchar color;

		vector<Point> contour(frameDiff.rows*2);

		for(int i = 0;i < frameDiff.rows;i++){

			for(int j = 0;j < frameDiff.cols;j++){
				
				color = frameDiff.at<uchar>(Point(j,i));

				if(color == 255){
					contour[i].x = j;
					contour[i].y = i;
					break;
				}
			}

			for(int j = frameDiff.cols-1;j >= 0;j--){
				color = frameDiff.at<uchar>(Point(j,i));
			
				if(color == 255){
					contour[i+frameDiff.rows].x = j;
					contour[i+frameDiff.rows].y = i;
					break;
				}
			}

		}
	}
    return 0;
}*/
/*
Mat *myCapture(VideoCapture cap){
	Mat *image;

	image = new Mat;

	cap.read(*image);
	return image;
}

void clearImage(Mat image){
	Vec3b color(0,0,0);
	for(int i = 0;i < image.rows;i ++){
		for(int j = 0;j < image.cols;j++){
			image.at<Vec3b>(Point(j,i)) = color;
		}
	}
}
/*
void pontero(bool* x){
	*x = true;
}

int main(){

	bool x;
	int j = 0;

	for(;;j++){
		if(j%2 == 0){
			pontero(&x);
		}
		printf("%d",x);
		x = false;
	}
}*/

#include <vector>       // std::vector
/*
void plusRGB(Mat *image){

	for( int y = 0; y < image->rows; y++ ){
		for( int x = 0; x < image->cols; x++ ){
			
			if(image->at<Vec3b>(y,x)[0] > 200 && image->at<Vec3b>(y,x)[1] > 200 && image->at<Vec3b>(y,x)[2] > 200){
				for( int c = 0; c < 3; c++ ){
					image->at<Vec3b>(y,x)[c] = (image->at<Vec3b>(y,x)[c] - 0.2*image->at<Vec3b>(y,x)[c]);	
				}
			}
			else{
				image->at<Vec3b>(y,x)[0] = (image->at<Vec3b>(y,x)[0] + 0.033*image->at<Vec3b>(y,x)[0]);	
				image->at<Vec3b>(y,x)[1] = (image->at<Vec3b>(y,x)[1] + 0.066*image->at<Vec3b>(y,x)[1]);
				image->at<Vec3b>(y,x)[2] = (image->at<Vec3b>(y,x)[2] + 0.022*image->at<Vec3b>(y,x)[2]);
			}
		}
	}

}

int main(int argc, char** argv)
{
    // READ RGB color image and convert it to Lab
    
	VideoCapture cap(0);
	Mat *image;
	cap.set(CV_CAP_PROP_HUE,8); //HUE 8
	cap.set(CV_CAP_PROP_SATURATION,93); //saturation 93
	for(int i =0;i<256;i++){
		image = myCapture(cap);
		imshow("",*image);
		printf("%d\n",i);
		waitKey(20);
	}

	/*cv::Mat bgr_image;
    cv::Mat lab_image;
	BackgroundSubtractorMOG2 bg;

	bg.nmixtures  =3;
	bg.bShadowDetection = false;

	int erosion_size = 3;
    Mat element = getStructuringElement(cv::MORPH_CROSS,
                                        cv::Size(erosion_size +1 ,erosion_size + 1),
                                        cv::Point(erosion_size, erosion_size) );

	while(1){
		cap.read(bgr_image);
		//plusRGB(&bgr_image);
		cv::cvtColor(bgr_image, lab_image, CV_BGR2Lab);

		// Extract the L channel
		std::vector<cv::Mat> lab_planes(4);
		cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

		// apply the CLAHE algorithm to the L channel
		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
		clahe->setClipLimit(3);
		cv::Mat dst;
		clahe->apply(lab_planes[0], dst);

		// Merge the the color planes back into an Lab image
		dst.copyTo(lab_planes[0]);
		cv::merge(lab_planes, lab_image);

	   // convert back to RGB
	   cv::Mat image_clahe;
	   cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);

	   bg.operator ()(bgr_image,lab_image);
	   cv::erode(lab_image,lab_image,element);
	   cv::dilate(lab_image,lab_image,element);	

	   // display the results  (you might also want to see lab_planes[0] before and after).
	   cv::imshow("image original", lab_image);
	   cv::imshow("image CLAHE", image_clahe);
	   cv::waitKey(5);
	}*//*
	return 0;
}
*/
double alpha; /**< Simple contrast control */
int beta;  /**< Simple brightness control */
 /*
void teste(void* arg){
	printf("Sera que funciona essa bagaca?");
	_sleep(10);
}



int main( int argc, char** argv )
{
 VideoCapture cap(0);


Mat image;
 for(;;){
	cap.read(image);
	 
	plusRGB(&image);

	imshow("Original Image", image);
	waitKey(5);
 }
return 0;
}
*/
int main(int argc, char *argv[])
{
    cv::Mat frame;                                              
    cv::Mat fg;     
    cv::Mat blurred;
    cv::Mat thresholded;
    cv::Mat gray;
    cv::Mat blob;
    cv::Mat bgmodel;                                            
    cv::namedWindow("Frame");   
    cv::namedWindow("Background Model");
    cv::namedWindow("Blob");
    cv::VideoCapture cap(0);    
cv::BackgroundSubtractorMOG2 bgs;                           

    bgs.nmixtures = 3;
    bgs.history = 1000;
    bgs.varThresholdGen = 15;
    bgs.bShadowDetection = true;                            
    bgs.nShadowDetection = 0;                               
    bgs.fTau = 0.5;                                         

std::vector<std::vector<cv::Point>> contours;               

for(;;)
{
    cap >> frame;                                           

    cv::GaussianBlur(frame,blurred,cv::Size(3,3),0,0,cv::BORDER_DEFAULT);

    bgs.operator()(blurred,fg);                         
    bgs.getBackgroundImage(bgmodel);                                

    cv::erode(fg,fg,cv::Mat(),cv::Point(-1,-1),1);                         
    cv::dilate(fg,fg,cv::Mat(),cv::Point(-1,-1),3);       

    cv::threshold(fg,thresholded,70.0f,255,CV_THRESH_BINARY);

    cv::findContours(thresholded,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
    cv::cvtColor(thresholded,blob,CV_GRAY2RGB);
    cv::drawContours(blob,contours,-1,cv::Scalar(255,255,255),CV_FILLED,8);

    cv::cvtColor(frame,gray,CV_RGB2GRAY);
    cv::equalizeHist(gray, gray);

    int cmin = 20; 
    int cmax = 1000;
    std::vector<cv::Rect> rects;
    std::vector<std::vector<cv::Point>>::iterator itc=contours.begin();
    while (itc!=contours.end()) {   
        if (itc->size() > cmin && itc->size() < cmax){ 

                    std::vector<cv::Point> pts = *itc;
                    cv::Mat pointsMatrix = cv::Mat(pts);
                    cv::Scalar color( 0, 255, 0 );

                    cv::Rect r0= cv::boundingRect(pointsMatrix);
                    cv::rectangle(frame,r0,color,2);                    

                    //DETECT THE DIRECTION OF MOVING OBJECTS HERE!

                    ++itc;}
         else{++itc;}
    }

    cv::imshow("Frame",frame);
    cv::imshow("Background Model",bgmodel);
    cv::imshow("Blob",blob);
    if(cv::waitKey(30) >= 0) break;
}
    return 0;
}
	