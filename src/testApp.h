#pragma once

#include "ofMain.h"
#include "ofxOpenCv.h"
//#include "../../../../addons/ofxOpenCv/libs/opencv/include/opencv2/imgproc/imgproc_c.h"

class testApp : public ofBaseApp{
	public:
		void setup();
		void update();
		void draw();
		
		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y);
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
        ofVideoGrabber 		vidGrabber;
        ofxCvColorImage			colorImg;
    
        ofxCvGrayscaleImage 	grayImage;
    
        IplImage * TheInput;
        IplImage * GrayCV;
        CvSURFParams params;
      
        ofImage PatternRead;
        ofxCvColorImage	PatternColor;
        ofxCvGrayscaleImage PatternGray;
        IplImage * PatternCV;
        cv::Mat Outmatch;
        int G_TechNiSel;
       cv::Mat G_tempoaver;
       cv::Mat G_DestinyForWarp;
       int G_DrawMode;
       // to get the points of the input
       vector<cv::KeyPoint> G_keypoints1;
    
    
  };
