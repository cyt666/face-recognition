 #include "opencv2/objdetect/objdetect.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"
 #include "opencv2/core/core.hpp"
 #include "opencv2/contrib/contrib.hpp"
 
 #include <iostream>
 #include <stdio.h>
 #include <fstream>
 #include <string.h>
 using namespace std;
 using namespace cv;

 int count = 0,facesSaved = 0;
 /** Function Headers */
 void detectAndDisplay( Mat frame, bool flag, Ptr<FaceRecognizer> model, vector<string> labelNames );
 std::vector<Rect> faces;
 /** Global variables */
 String face_cascade_name = "haarcascades/haarcascade_frontalface_alt.xml";
 CascadeClassifier face_cascade;
 string window_name = "Capture - Face detection";
 RNG rng(12345);
 
 int detectCount[3]={0,0,0};
 string faceToDetect;
 enum Mode {LBPH,EIGEN};
 Mode mode = LBPH ;//mode of operation
 bool DEBUG = true;
 /** @function main */
 int main( int argc, const char** argv )
 {
   CvCapture* capture;
   Mat frame;
   //-- 1. Load the cascades
   if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
   if(argc < 2)
   {
	   cout << "usage: " << argv[0] << " <csv.ext> <Mode:LBPH | EIGEN>" << endl;
	   cout<< "please give the csv file you are using for this program."<<endl;
       exit(1);
   }
   if (argc == 3) 
   {
		if(!strcmp(argv[2],"EIGEN"))
		{
			mode = EIGEN ;
		}
	}
   if (mode == LBPH)
		cout<<"running in LBPH mode\n";
   else
		cout<<"running in eigen mode\n";
	if(DEBUG)
	{
		cout<<"DEBUG MODE active !!! \n";
		char ch;
		cout<<"continue in DEBUG mode? (y/n)";
		cin>>ch;
		if (ch == 'n' || ch == 'n')
		{
			DEBUG = false;
		}
	}
	if(DEBUG)
	{
		cout<<"Enter the name of the person to be tested (lowercase):";
		cin >>faceToDetect;
	}		
   //-- 2. Read the video stream
   capture = cvCaptureFromCAM( -1 );
   Ptr<FaceRecognizer> model ;
   if (mode == LBPH)
		model = createLBPHFaceRecognizer(1,8,8,8,80.0);
   else
		model = createEigenFaceRecognizer();
   string csv_file  = argv[1];
   if (mode == LBPH)
		model->load(csv_file+"_LBPH.yml");
   else
		model->load(csv_file+"_eigen.yml");
   string dataIn = csv_file+".dat";
   std::ifstream file(dataIn.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    vector<string> labelNames;
    string line;
    while (getline(file, line)) {
		labelNames.push_back(line);
	}
	bool faceInDatabase = false;
	for(int i = 0 ; i < labelNames.size() ; i++)
	{
		if(labelNames[i] == faceToDetect)
		{
			faceInDatabase = true;
			break;
		}
	}
	if(!faceInDatabase)
	{
		cout<<"person not in database\n";
		faceToDetect = faceToDetect + "(not_in_database)";
	}
			
   if( capture )
   {
     while( true )
     {
		  bool flag = false; //whether to take images of faces
		  frame = cvQueryFrame( capture );
		  int c = waitKey(2);
		  if( (int)c == 27 ) { break; }
		  else
			if((int)c == 73 || (int)c==105)
			{
				flag = true;
			}
		      //-- 3. Apply the classifier to the frame
          if( !frame.empty() )
		   { detectAndDisplay( frame , flag, model, labelNames); }
		  else
		   { printf(" --(!) No captured frame -- Break!"); break; }
      }
   }
   if(DEBUG)
   {
	   int totalCount = detectCount[0]+detectCount[1]+detectCount[2];
	   cout << "for " << faceToDetect << ": TotalCount: " << totalCount << " undetected:" << detectCount[0]*100/totalCount << "% detected incorrectly:" 
	   << detectCount[1] *100/totalCount<< "% detected correctly:" << detectCount[2] *100/totalCount<< "%" << endl;
	   
	   std::ofstream fileOut;
	   fileOut.open("FaceRecoData.dat", ios::out | ios::app);
	   fileOut << faceToDetect << "\t" << totalCount << '\t' << detectCount[0]*100/totalCount << "%\t" << detectCount[1] *100/totalCount << "%\t" <<
	   detectCount[2] *100/totalCount <<"%\n";
	   fileOut.close();
   }
   return 0;
 }

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame , bool flag, Ptr<FaceRecognizer> model, vector<string> labelNames)
{
  extern std::vector<Rect> faces;
  Mat frame_gray;
  extern int count,facesSaved;
  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  if(count % 5 == 0){
	  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	  count = 0;
  }
  count++;

  for( size_t i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    Rect myROI(faces[i].x,faces[i].y,faces[i].width,faces[i].height);
    rectangle( frame, myROI, Scalar( 0, 255, 0 ), 4, 8, 0 );
	Mat croppedImage;
	Mat(frame,myROI).copyTo(croppedImage);
	resize(croppedImage,croppedImage,Size(200,200),0,0,CV_INTER_AREA);
	Mat greyCroppedImage;
	cvtColor(croppedImage, greyCroppedImage, CV_RGB2GRAY);
	int predictedLabel = -1;
    double confidence = 0.0;
    model->predict(greyCroppedImage	, predictedLabel, confidence);
    stringstream ss;
	ss << confidence;
	string conf_str = ss.str();
	//cout<<predictedLabel<<"\t"<<confidence;
	if(predictedLabel==-1)
	{
		detectCount[0]++;
	}
	else
	{
		if (DEBUG)
		{
			if(labelNames[predictedLabel]!=faceToDetect)
			{
				detectCount[1]++;
			}
			else
			{
				detectCount[2]++;
			}
		}
		string faceLabelString = "Detected: "+labelNames[predictedLabel];
		string confLabelString = "with confidence  " + conf_str;
		char const* confLabel = confLabelString.c_str(); 
		char const* faceLabel = faceLabelString.c_str();
		putText(frame,faceLabel,Point(faces[i].x,faces[i].y-30),2,0.7,Scalar(255,0,255 ));
		putText(frame,confLabel,Point(faces[i].x,faces[i].y-10),2,0.7,Scalar(255,0,255 ));
	}
		
	if(flag)
	{
	  stringstream ss;
	  srand(time(NULL));
	  ss << rand() % 100;
	  string str = ss.str();
	  string fileName = "myfaces/temp/myimage" + str +".png";
	  string savedInfoString = "face saved to " + fileName ;
	  cout<< savedInfoString <<endl;
	  imwrite(fileName,croppedImage);
	  facesSaved += 1;
    }
  }
  //-- Show what you got
  imshow( window_name, frame );
 }
