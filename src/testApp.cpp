#include "testApp.h"

#define Nx 640
#define Ny 480



  //cv::FastFeatureDetector detector(15);
bool G_PrintKeypoint;
//--------------------------------------------------------------
void testApp::setup(){
    
    vidGrabber.setVerbose(true);
    vidGrabber.initGrabber(Nx,Ny);
    colorImg.allocate(Nx,Ny);
	grayImage.allocate(Nx,Ny);
     TheInput = cvCreateImage(cvSize(Nx, Ny), 8, 3);
    GrayCV = cvCreateImage(cvSize(Nx, Ny), 8, 1);
    
  // PatternRead.loadImage("UAAZ2small.png");
    PatternRead.loadImage("patternbunny2.jpg");
    PatternColor.allocate(PatternRead.width, PatternRead.height);
    PatternColor.setFromPixels(PatternRead.getPixels(),PatternRead.width,PatternRead.height);
    PatternGray.allocate(PatternRead.width, PatternRead.height);
    PatternGray = PatternColor;
    PatternCV = cvCreateImage(cvSize(PatternRead.width, PatternRead.height), 8, 1);
    PatternCV = PatternGray.getCvImage();
    
    // blurring the template
    
   // cvSmooth(PatternCV, PatternCV,CV_BLUR,5);
    
    
    
    G_TechNiSel =0;
    G_PrintKeypoint = false;
    // Surf --> 0;
    
   // G_DestinyForWarp  = cvCreateImage(cvSize(400, 400), 8, 3);
    G_DrawMode = 0; // draw template ,image, matchings and unwarp
    
}

//--------------------------------------------------------------
void testApp::update(){
    bool bNewFrame = false;
    
    
    vidGrabber.update();
    bNewFrame = vidGrabber.isFrameNew();
    
	if (bNewFrame){
        
        
        colorImg.setFromPixels(vidGrabber.getPixels(), Nx,Ny);
        grayImage = colorImg;
        TheInput = colorImg.getCvImage();
        GrayCV = grayImage.getCvImage();

        
        // Code just for detection
//        //Storage
//        CvMemStorage* storage = cvCreateMemStorage(0);
//        //Define sequence for storing surf keypoints and descriptors
//        CvSeq *imageKeypoints = 0, *imageDescriptors = 0;
//        //Extract SURF points by initializing parameters
//        CvSURFParams params = cvSURFParams(500, 1);
//        cvExtractSURF( GrayCV, 0, &imageKeypoints, &imageDescriptors, storage, params );
//        
//        // draw surf points over the image
//        //draw the keypoints on the captured frame
//        for( int i = 0; i < imageKeypoints->total; i++ )
//        {
//            CvSURFPoint* r = (CvSURFPoint*)cvGetSeqElem( imageKeypoints, i );
//            CvPoint center;
//            int radius;
//            center.x = cvRound(r->pt.x);
//            center.y = cvRound(r->pt.y);
//            radius = cvRound(r->size*1.2/9.*2);
//            cvCircle( TheInput, center, radius, CV_RGB(255, 0, 0), 1, 8, 0 );
//        }
        
        // surft
        cv::SurfFeatureDetector detectorSURF(1500);
        cv::SurfDescriptorExtractor extractorSURF;

        //sift
        cv::SiftFeatureDetector detectorSIFT(.05,5.0);
        cv::SiftDescriptorExtractor extractorSIFT(3.0);
        
        
        
        cv::Mat descriptors1, descriptors2;
        
        vector<cv::KeyPoint>  keypoints2;
        G_keypoints1.clear();
        switch (G_TechNiSel) {
            case 0:
                detectorSURF.detect(PatternCV, G_keypoints1);
                detectorSURF.detect(GrayCV, keypoints2);
                extractorSURF.compute(PatternCV, G_keypoints1, descriptors1);
                extractorSURF.compute(GrayCV, keypoints2, descriptors2);
                
                break;
            case 1:
                detectorSIFT.detect(PatternCV, G_keypoints1);
                detectorSIFT.detect(GrayCV, keypoints2);
                extractorSIFT.compute(PatternCV, G_keypoints1, descriptors1);
                extractorSIFT.compute(GrayCV, keypoints2, descriptors2);
                break;
            default:
                break;
        }
        
        G_tempoaver = descriptors1;

     //   IplImage TempoAver = descriptors1;
        double TheMin=0;
        double TheMax=0;
        cv::minMaxLoc(descriptors1, &TheMin, &TheMax);
        if(G_PrintKeypoint==true){
            
            cout<<"Minimun:"<<TheMin<<"\n";
            cout<<"Maximun:"<<TheMax<<"\n";
  //            for (int k=0; k<keypoints1.size(); k++) {
//              
//                cout<<"Id:"<<keypoints1[k].class_id<<"\n";
//                cout<<"Octave:"<<keypoints1[k].octave<<"\n";
//                cout<<"X:"<<keypoints1[k].pt.x<<"\n";
//                cout<<"y:"<<keypoints1[k].pt.y<<"\n";
//                cout<<"Response:"<<keypoints1[k].response<<"\n";
//                cout<<"Size:"<<keypoints1[k].size<<"\n";
//            }
            G_PrintKeypoint= false;
        }
       // cv::BruteForceMatcher<cv::L2<float> > matcher;

        cv::FlannBasedMatcher matcher;
        vector<cv::DMatch> matches;
        matcher.match(descriptors1, descriptors2, matches);
        
         double max_dist = 0; double min_dist = 100;
        
        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptors1.rows; i++ )
        { double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }
        
        vector< cv::DMatch > good_matches;
        
        for( int i = 0; i < descriptors1.rows; i++ )
        { if( matches[i].distance < 5*min_dist )
        { good_matches.push_back( matches[i]); }
        }

        drawMatches( PatternCV, G_keypoints1, GrayCV, keypoints2,
                    good_matches, Outmatch, cvScalarAll(-1), cvScalarAll(-1),
                    vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        
        
        //-- Localize the object
        vector<cv::Point2f> obj;
        vector<cv::Point2f> scene;
        
        for( int i = 0; i < good_matches.size(); i++ )
        {
            //-- Get the keypoints from the good matches
            obj.push_back( G_keypoints1[ good_matches[i].queryIdx ].pt );
            scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
        }
        
        
        if (good_matches.size()>=4) {
  
        cv::Mat H = findHomography( obj, scene, CV_RANSAC  );
        
        //-- Get the corners from the image_1 ( the object to be "detected" )
        vector<cv::Point2f> obj_corners(4);
        obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( PatternCV->width, 0 );
        obj_corners[2] = cvPoint( PatternCV->width, PatternCV->height ); 
        obj_corners[3] = cvPoint( 0, PatternCV->height );
        vector<cv::Point2f> scene_corners(4);
        
        perspectiveTransform( obj_corners, scene_corners, H);
        
        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( Outmatch, scene_corners[0] + cv::Point2f( PatternCV->width, 0),
             scene_corners[1] + cv::Point2f( PatternCV->width, 0), cvScalar(0, 255, 0), 4 );
        line( Outmatch, scene_corners[1] + cv::Point2f(PatternCV->width, 0), 
             scene_corners[2] + cv::Point2f( PatternCV->width, 0), cvScalar( 0, 255, 0), 4 );
        line( Outmatch, scene_corners[2] + cv::Point2f( PatternCV->width, 0)
             , scene_corners[3] + cv::Point2f( PatternCV->width, 0), cvScalar( 0, 255, 0), 4 );
        line( Outmatch, scene_corners[3] + cv::Point2f( PatternCV->width, 0),
             scene_corners[0] + cv::Point2f( PatternCV->width, 0), cvScalar( 0, 255, 0), 4 );

        
     //   drawMatches(PatternCV, keypoints1, GrayCV, keypoints2, matches, Outmatch);
//          cout<<"cols:"<<Outmatch.cols<<"\n";
//          cout<<"rows:"<<Outmatch.rows<<"\n";
//          cout<<"dims:"<<Outmatch.dims<<"\n";

            // Warping the Image Back!
            
            // Destination Points
               vector<cv::Point2f> Rewarp_corners(4);
            Rewarp_corners[0] = cvPoint(350,350);
            Rewarp_corners[1] = cvPoint(450,350);
            Rewarp_corners[2] = cvPoint(450,450);
            Rewarp_corners[3] = cvPoint(350,450);
            
            
            cv::Mat newH =findHomography( scene_corners, Rewarp_corners, CV_RANSAC );
         
            // the source image
            
            cv::Mat InputMAT(TheInput);
            cv::warpPerspective(InputMAT, G_DestinyForWarp, newH, cvSize(800, 800));
            
            //cvWarpPerspective(TheInput, G_DestinyForWarp, &newH);
            
            
        }
    }

}

//--------------------------------------------------------------
void testApp::draw(){
	ofSetHexColor(0xffffff);
    
    if (G_DrawMode==0) {

        ofxCvColorImage AuxGrayImage;
        AuxGrayImage.allocate(Outmatch.cols, Outmatch.rows);
        AuxGrayImage = Outmatch.data;
        AuxGrayImage.draw(0, 0);
    
    
        ofxCvColorImage ReWarp;
        ReWarp.allocate(G_DestinyForWarp.cols, G_DestinyForWarp.rows);
        ReWarp = G_DestinyForWarp.data;
        ReWarp.draw(Outmatch.cols,0,800,800);
    }
    if (G_DrawMode==1) {
        ofxCvColorImage ReWarp;
        ReWarp.allocate(G_DestinyForWarp.cols, G_DestinyForWarp.rows);
        ReWarp = G_DestinyForWarp.data;
        ReWarp.draw(ofGetWidth()/2.0-G_DestinyForWarp.cols/2.0,0,800,800);
    
    
    }
    if (G_DrawMode==2) {
        
        PatternRead.draw(ofGetWidth()/2.0-PatternRead.width/2.0,100);
       
   
        cv::Mat Otra;
        cv::Mat OtraMas;
        
        cv::transpose(G_tempoaver, Otra);
        
        double minVal, maxVal;
        minMaxLoc(Otra, &minVal, &maxVal); //find minimum and maximum intensities
    
      //  Otra.convertTo(OtraMas, CV_8U);
        
        Otra.convertTo(OtraMas, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
        //  cv::transpose(Otra, Otra);
        ofxCvGrayscaleImage GrayTemp2;
        GrayTemp2.allocate(OtraMas.cols, OtraMas.rows);
        GrayTemp2 = OtraMas.data;
       // GrayTemp2.transform(90, GrayTemp2.width/2.0, GrayTemp2.height/2.0, 1.0, 1.0, 0.0, 0.0);
        GrayTemp2.draw(ofGetWidth()/2.0-4*OtraMas.cols/2.0, 100+PatternRead.height+100,
                       OtraMas.cols*4,OtraMas.rows*4);
        
        float Xoff1 = ofGetWidth()/2.0-PatternRead.width/2.0;
        float Xoff2 = ofGetWidth()/2.0-4*OtraMas.cols/2.0;
        
        // ploting lines
        ofColor TheColor;
        ofSetLineWidth(2);
        for (int k = 0; k < G_keypoints1.size(); k++) {
            //255*(float)(k/G_keypoints1.size())
            TheColor.setHsb(255*(k/(float)G_keypoints1.size()), 128, 128);
            ofNoFill();
            ofCircle(G_keypoints1[k].pt.x+Xoff1, G_keypoints1[k].pt.y+100, 4);
             ofSetColor(TheColor);
            ofLine(G_keypoints1[k].pt.x+Xoff1, G_keypoints1[k].pt.y+100,
                   Xoff2+4*k, 100+PatternRead.height+100);

        }
        
    }
    
    

    
}

//--------------------------------------------------------------
void testApp::keyPressed(int key){

    switch (key) {
        case 'm':
            G_TechNiSel++;
            if (G_TechNiSel>1){G_TechNiSel=0;}
            break;
     
        case 'p':
            G_PrintKeypoint = true;
            break;
            
      
    case 'd':
            G_DrawMode++;
            if (G_DrawMode>2){G_DrawMode=0;}
            break;
              default:
            break;
    }
    
    
}

//--------------------------------------------------------------
void testApp::keyReleased(int key){

}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y){

}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void testApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void testApp::dragEvent(ofDragInfo dragInfo){ 

}