#include <opencv2\opencv.hpp>
#include <process.h>
#include <vector> 
#include <iostream>
#include <windows.h>

using namespace cv;
using namespace std;

#pragma region testes malditos
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

     // std::vector
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
*//*
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
*/
/*
Codigos antigos




	
		//movement = searchForMovement(fore,cameraFeed,movement,&moveDetect);
		
		//if(!moveDetect)
			//movement = clearImage();

*/
/*
vector<vector<Point>> getContours(Mat fore){
		vector<vector<Point>>contours;
				cv::findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
				return contours;

}

int main(){

	
	// background detection properties
	cv::BackgroundSubtractorMOG2 bg;
    bg.nmixtures = 3;
    bg.bShadowDetection = false;
	VideoCapture cap(0);
	Mat image,fore;
	vector<vector<Point>>contours;

	for(;;){

		cap.read(image);
		bg.operator ()(image,fore);
		 cv::erode(fore,fore,cv::Mat(),cv::Point(-1,-1),1);                         
		cv::dilate(fore,fore,cv::Mat(),cv::Point(-1,-1),3); 	
		
		contours = getContours(fore); 
		cv::drawContours(image,contours,-1,cv::Scalar(255,255,255),CV_FILLED,8);


		imshow("",image);
		waitKey(5);
	}
	return 0;
}*/

/*
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
typedef enum{
    MULHER,
    HOMEM,
	JAPONES
}genero;
int chekingGender(float, float);

static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
        case 1:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
            break;
        case 3:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
            break;
        default:
            src.copyTo(dst);
            break;
    }
    return dst;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int chekingGender(float height, float width) {
    int x = 0;
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    
    string output_folder = ".";
    
    // Get the path to your CSV.
    string fn_csv = string("C:\\opencv\\people.csv");
    string fn_haar = string("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt_tree.xml");
    // These vectors hold the images and coresponding labels.
    vector<Mat> images;
    vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
	
    int im_width = images[0].cols;
    int im_height = images[0].rows;
    // Quit if there are not enough images for this demo.
    
	images.size();

	if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size:
    Mat testSample = images[images.size() - 1];
    images.pop_back();
    labels.pop_back();
    // The following lines create an Fisherfaces model for
    // face recognition and train it with the images and
    // labels read from the given CSV file.
    // If you just want to keep 10 Fisherfaces, then call
    // the factory method like this:
    //
    //      cv::createFisherFaceRecognizer(10);
    //
    // However it is not useful to discard Fisherfaces! Please
    // always try to use _all_ available Fisherfaces for
    // classification.
    //
    // If you want to create a FaceRecognizer with a
    // confidence threshold (e.g. 123.0) and use _all_
    // Fisherfaces, then call it with:
    //
    cv::createFisherFaceRecognizer(0, 123.0);
    //
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    model->train(images, labels);
    // The following line predicts the label of a given
    // test image:
    
    int predictedLabel = -1;
    double confidence = 0.0;
    model->predict(testSample, predictedLabel, confidence);
    
    // Here is how to get the eigenvalues of this Eigenfaces model:
    Mat eigenvalues = model->getMat("eigenvalues");
    // And we can do the same to display the Eigenvectors (read Eigenfaces):
    Mat W = model->getMat("eigenvectors");
    // Get the sample mean from the training data
    Mat mean = model->getMat("mean");
    
    
    // That's it for learning the Face Recognition model. You now
    // need to create the classifier for the task of Face Detection.
    // We are going to use the haar cascade you have specified in the
    // command line arguments:
    //
    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);
    // Get a handle to the Video device:
    VideoCapture cap(0);
    Mat frame;
    cap >> frame;
    cap.set(CV_CAP_PROP_FRAME_WIDTH, (frame.cols/2));
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, (frame.rows/2));
	cap.set(CV_CAP_PROP_HUE,8); //HUE 8
	cap.set(CV_CAP_PROP_SATURATION,93); //saturation 93

    //cap.set(CV_CAP_PROP_FRAME_WIDTH, (width));
    //cap.set(CV_CAP_PROP_FRAME_HEIGHT, (height));
    //printf("width %f, height %f", width, height);
    
    // Check if we can use this device at all:
    if(!cap.isOpened()) {
        return -1;
    }
    // Holds the current frame from the Video device:
    for(;;) {
		x++;
		stringstream ss ;
		ss << "C:\\opencv\\helder\\helder" <<x<< ".png";
		string s = ss.str();
        cap >> frame;
        Mat original = frame.clone();
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        // Find the faces in the frame:
        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces);
        for(int i = 0; i < faces.size(); i++) {
            // Process face by face:
            Rect face_i = faces[i];
            // Crop the face from the image. So simple with OpenCV C++:
			
            Mat face = gray(face_i);
			Mat rgb = original(face_i);
			resize(rgb,rgb,Size(200,200));

			imwrite(s,rgb);
            Mat face_resized;
            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
			
			// Now perform the prediction, see how easy that is:
            int prediction = model->predict(face_resized);
            // And finally write all we've found out to the original image!
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
            string box_text;
			printf("%d ",prediction);
            if(prediction == MULHER){
                box_text = format("Prd - 0");
			}
			else
				box_text = format("Prd - 1");
		

            // Calculate the position for annotated text (make sure we don't
            // put illegal values in there):
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            // And now put it into the image:
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 2.0);
        }
        // Show the result:
        imshow("Gender Recognizer - Joao Lucas Sisanoski", original);
        
        // And display it:
        char key = (char) waitKey(20);
        // Exit this loop on escape:
        if(key == 27)
            break;
    }
    
    return 0;
}

*/
#pragma endregion

	typedef struct dataStruct {
    int ID;
    int returnValue;
	};



void teste(void* param){
	
	dataStruct *thread = (dataStruct*)param;
	Sleep(500);
	thread->ID = 500;
	thread->returnValue = 50;

	
	printf("ID: %d\nReturned: %d\n",thread->ID,thread->returnValue);
}

void teste2(void* param){
	
	dataStruct *thread = (dataStruct*)param;
	Sleep(100);
	thread->ID = 100;
	thread->returnValue = 10;

	
	printf("ID: %d\nReturned: %d\n",thread->ID,thread->returnValue);
}

int main(){
	
	
	dataStruct thread_1,data2;

	_beginthread( teste, 0, (void*)&thread_1 );
	_beginthread( teste2, 0, (void*)&thread_1 );


	for(;;){
	}
	
	return 0;
}

