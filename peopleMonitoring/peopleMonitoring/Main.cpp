#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>



using namespace cv;
using namespace std;

int im_width,im_height;
int intrestingPeople = 0;
int numberOfPeople = 0;

//Define the standard window size
const static int windowHeight = 288, windowWidth = 352;
//our sensitivity value to be used in the absdiff() function
const static int SENSITIVITY_VALUE = 20;
//size of blur used to smooth the intensity image output from absdiff() function
const static int BLUR_SIZE = 10;


typedef enum{
    MULHER,
    HOMEM
}genero;

class skindetector
{
public:
    skindetector(void);
    ~skindetector(void);
    
    cv::Mat getSkin(cv::Mat input);
    
private:
    int Y_MIN;
    int Y_MAX;
    int Cr_MIN;   
	int Cr_MAX;
    int Cb_MIN;
    int Cb_MAX;
};skindetector::skindetector(void)
{
    //YCrCb threshold
    // Range of skin color values
    Y_MIN  = 0;
    Y_MAX  = 255;
    Cr_MIN = 133;
    Cr_MAX = 170;
    Cb_MIN = 75;
    Cb_MAX = 135;
}

skindetector::~skindetector(void)
{
}

Mat skindetector::getSkin(cv::Mat input)
{
    Mat skin;
    //first convert our RGB image to YCrCb
    cvtColor(input,skin,cv::COLOR_BGR2YCrCb);
    
    //filter the image in YCrCb color space
    inRange(skin,cv::Scalar(Y_MIN,Cr_MIN,Cb_MIN),cv::Scalar(Y_MAX,Cr_MAX,Cb_MAX),skin);
    return skin;
}

Mat getImagePart(Mat image,Rect partRect){
	
	// Create an image with the size of rect received
	Mat imagePart(partRect.height,partRect.width,CV_8UC3);
	// Temp variable to copy the pixel value
    Vec3b color;

	for(int i = 0 ; i < imagePart.rows;i++){
		for(int j = 0;j < imagePart.cols;j++){
			// Copy the pixel value of the original image, and put it in the new image with new rect
			color = image.at<Vec3b>(Point(j+partRect.x,i+partRect.y));
			imagePart.at<Vec3b>(Point(j,i)) = color;
		}
	}
	return imagePart;
}

void initFunctions(KalmanFilter KF){
	KF.statePre.at<float>(0) = 0;
    KF.statePre.at<float>(1) = 0;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;

    KF.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1); // Including velocity
    KF.processNoiseCov = *(cv::Mat_<float>(4,4) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,  0,0,0,0.3);

    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KF.errorCovPost, Scalar::all(.1));
}

void initRectSize(int x, int y, int width, int height,Rect *rect){
	//initialize the rect values
	rect->x = x;
	rect->y = y;
	rect->width = width;
	rect->height = height;
}

void getSkinMat(Mat frame,Mat *outImage){
	skindetector mySkinDetector;
	vector<vector<Point>> *contours = new vector<vector<Point>>;
	vector<Vec4i> hierarchy;
	RNG rng;

	int erosion_size = 2;
    Mat element = getStructuringElement(cv::MORPH_CROSS,
                                        cv::Size(erosion_size +1 ,erosion_size + 1),
                                        cv::Point(erosion_size, erosion_size) );
    

	Scalar color = Scalar( rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255) );

	*outImage = mySkinDetector.getSkin(frame);

	cv::erode(*outImage, *outImage, element);
	cv::dilate(*outImage, *outImage, element);
	
}

vector<vector<Point>> getContours(Mat image,int* x){
	vector<vector<Point>> contours;
	vector<Vec4i>hierarchy;
	double contourArea = 0;
	// Search the contours in the image and put it in a Matrix of points.
	findContours(image,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,Point(0,0)); //Get Skin contours

		for( int i = 0; i< (int)contours.size(); i++) {
			if(cv::contourArea(contours[i]) > 300){
				if(cv::contourArea(contours[i]) > contourArea){
					contourArea = cv::contourArea(contours[i]);
					*x = i;
				}
				drawContours( image, contours, i,Scalar(127,127,127), 1.5, 1, hierarchy, CV_16SC1, Point() );
			}
		}
		return contours;
}


void initCapture(VideoCapture* capture){
	capture->open(0);
	capture->set(CV_CAP_PROP_FRAME_WIDTH,windowWidth);
	capture->set(CV_CAP_PROP_FRAME_HEIGHT,windowHeight);
	capture->set(CV_CAP_PROP_HUE,8); //HUE 8
	capture->set(CV_CAP_PROP_SATURATION,93); //saturation 93
}

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

Ptr<FaceRecognizer> chekingGender(CascadeClassifier* haar_cascade) {
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
	
    im_width = images[0].cols;
    im_height = images[0].rows;
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
    
  /*  // Here is how to get the eigenvalues of this Eigenfaces model:
    Mat eigenvalues = model->getMat("eigenvalues");
    // And we can do the same to display the Eigenvectors (read Eigenfaces):
    Mat W = model->getMat("eigenvectors");
    // Get the sample mean from the training data
    Mat mean = model->getMat("mean");
*/
    haar_cascade->load(fn_haar);

	return model;
}




int main(){
	
	// These pragmas is used to create a code block. It helps us to hide the code
	#pragma region init local objects
		VideoCapture capture;
		Mat cameraFeed;
		Mat leftSkinMat;
		Mat rightSkinMat;
		Mat leftFrame;
		Mat rightFrame;
		Mat skinMat;
		vector<vector<Point> > leftContours;
		vector<vector<Point>>rightContours;
		vector<Point2f> leftDirection(5);
		vector<Point2f> rightDirection(5);
		vector<Vec4i> leftHierarchy;
		vector<Vec4i> rightHierarchy;
		KalmanFilter KF(4, 2, 0);
		Rect leftR;
		Rect rightR;
		Rect window;
		Rect partRect;
		CascadeClassifier haar_cascade;
		Ptr<FaceRecognizer> model = chekingGender(&haar_cascade);
	#pragma endregion
	#pragma region init local variables
		bool lFlag = false;
		bool rFlag = false;
		bool flag = false;
		int lTemp = 0;
		int rTemp = 0;
		int lX = 0;
		int lDx = 0;
		int rX = 0; 
		int rDx = 0;
		int teste = 0;
		int temp = 0;
	#pragma endregion 

	//Init Kalman filter configurations
	initFunctions(KF);
	// init camera capture configuration
	initCapture(&capture);
	// init rect sizes
	initRectSize(0,0,windowWidth,windowHeight,&window); //Used to search the skins in the window.
	initRectSize(0,0,50,windowHeight,&leftR);//Used to detect people in the left of the window.
	initRectSize(windowWidth - 50,0,50,windowHeight,&rightR);//Used to detect people in the left of the window.



	for(;;) {
	
		//initialize the camera
		capture.read(cameraFeed);

		//Get the sides frames 
		leftFrame = getImagePart(cameraFeed,leftR);
		rightFrame = getImagePart(cameraFeed,rightR);

		//Get the skins in the rect of before frames.
		getSkinMat(leftFrame,&leftSkinMat);
		getSkinMat(rightFrame,&rightSkinMat);

		//Index of the bigger area contour. Used to count the number of people passed.
	    int j = -1;
		int k = -1;

		//Get the contours created by the skins in the rect.
		leftContours = getContours(leftSkinMat,&j);
		rightContours = getContours(rightSkinMat,&k);
		
		Moments lmu,rmu;	
		Point2f lmc = -1,rmc = -1;
		lDx = 0;
		rDx = 0;


		//printf("%d",j);

		if(j < 0)
			lTemp = 0;
		if(k < 0)
			rTemp = 0;


		if(j >= 0 && lTemp < 5){
			lmu = moments( leftContours[j], false );
			lmc = Point2f( lmu.m10/lmu.m00 , lmu.m01/lmu.m00 ); 
			leftDirection[lTemp] = Point2f( lmu.m10/lmu.m00 , lmu.m01/lmu.m00 );
			//drawCross(cameraFeed, mc, Scalar(255, 0, 0), 5);
			lTemp++;
		}

		if(k >= 0 && rTemp < 5){
			rmu = moments(rightContours[k],false);
			rmc = Point2f( rmu.m10/rmu.m00 , rmu.m01/rmu.m00 ); 
			rightDirection[rTemp] = Point2f( rmu.m10/rmu.m00 , rmu.m01/rmu.m00 );
			//drawCross(cameraFeed, mc, Scalar(255, 0, 0), 5);
			rTemp++;
		}
 

		if(lTemp == 5){
			lX = (int)leftDirection[0].x;
			for(int i = 1;i < (int)leftDirection.size();i++){
				lDx += (int)(lX - leftDirection[i].x); 
			}
		}
		if(rTemp == 5){
			rX = rightDirection[0].x;
			for(int i = 0;i < (int)rightDirection.size();i++){
				rDx += (int)(rX - rightDirection[i].x);
			}
		}

		//printf("ldx: %d jrdx: %d\n",lDx,rDx);

		if(lmc.inside(window) && lFlag == false){
			if(lDx < 0){
				numberOfPeople++;
				lFlag = true;
				printf("Total :%d\nIntresting: %d\n ",numberOfPeople,intrestingPeople);
			}
		}
		if(rmc.inside(window) && rFlag == false){
			if(rDx > 0){
				numberOfPeople++;
				rFlag = true;
				printf("Total :%d\nIntresting: %d\n ",numberOfPeople,intrestingPeople);

			}
		}

		if(!lmc.inside(window))
			lFlag = false;
		if(!rmc.inside(window))
			rFlag = false;


        Mat original = cameraFeed.clone();

		Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);

        // Find the faces in the frame:
        vector< Rect_<int> > faces;

        haar_cascade.detectMultiScale(gray, faces);
		temp = faces.size();

		if(faces.size() == 0 || temp-teste != 0)
			flag = true;

	#pragma region for faces < i
		for(int i = 0; i < faces.size(); i++) {
			if(flag == true){
				teste = faces.size();
				intrestingPeople += teste;
				//printf("%d ",teste);
			}
			flag = false;
			// Process face by face:
            Rect face_i = faces[i];
            // Crop the face from the image. So simple with OpenCV C++:
			
            Mat face = gray(face_i);
			
            Mat face_resized;
            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
			
			// Now perform the prediction, see how easy that is:
            int prediction = model->predict(face_resized);
            // And finally write all we've found out to the original image!
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
            string box_text;
			
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
	#pragma endregion

		
		
	
        // Show the result:
		/*
        imshow("People Monitoring", original);
        imshow("right",rightSkinMat);
		imshow("left",leftSkinMat);
		*/
        // And display it:
        waitKey(1);
    }

	return 0;
}