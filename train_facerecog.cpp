#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

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
    vector<string> labelNames;
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel, dirName, subDirName;
    while (getline(file, line)) 
    {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        stringstream pathstream(path);
        getline(pathstream,dirName,'/');
        getline(pathstream,subDirName,'/');
        //cout<<subDirName<<endl;
        if(!path.empty() && !classlabel.empty() && subDirName != "temp") 
        {
            images.push_back(imread(path, 0));
            int label = atoi(classlabel.c_str());
            labels.push_back(label);
            if(labelNames.size()<label+1)
            {
				labelNames.push_back(subDirName);
			}
        }
    }
    string fileOutName = filename+".dat";
    std::ofstream fileOut(fileOutName.c_str(),std::ofstream::out);
    for(int i=0;i<labelNames.size();i++){
		fileOut<<labelNames[i]<<endl;
	}
	fileOut.close();
}

enum Mode {LBPH,EIGEN};
Mode mode = LBPH ;//mode of operation

int main(int argc, const char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc < 2) {
        cout << "usage: " << argv[0] << " <csv_File.ext> <Mode:LBPH | EIGEN>" << endl;
        exit(1);
    }
    if (argc == 3) {
		if(!strcmp(argv[2],"EIGEN")){
			mode = EIGEN ;
			cout<<"eigen";
		}
	}
		
    // Get the path to your CSV.
    string fn_csv = string(argv[1]);
    // These vectors hold the images and corresponding labels.
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
    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size:
    int height = images[0].rows;
    if(mode == LBPH){
		Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
		model->train(images, labels);
		model->save(fn_csv+"_LBPH.yml");
		cout << "model saved successfully to " << fn_csv+"_LBPH.yml\n" ;
	}
	else{	
		Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
		model->train(images, labels);
		model->save(fn_csv+"_eigen.yml");
		Mat eigenvalues = model->getMat("eigenvalues");
		cout << "model saved successfully to " << fn_csv+"_eigen.yml\n" ;
	}
    return 0;
}
