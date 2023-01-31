#include"eventCls.h"
#include<stdio.h>
#include <sys/time.h>


using namespace std;

int main(int argc, char** argv)
{
    AlgEventCLS* clsprocessor = new AlgEventCLS();
    // read image
    cout << "read image ..." << endl;
    cv::Mat images= cv::imread("/home/zhaoming/data/site_data/32030013-2022_12_29_11_18_22_250003_OUTSIDE_OPERATIO_2.jpg");
    PredRes *results = new PredRes();
//    clsprocessor->getEvents(images,results);
    struct timeval detstart;
    struct timeval detend;
    gettimeofday(&detstart,NULL);
    clsprocessor->getEvents(images,results);
    std::cout<<"Event : "<<results->cls<<" ; "<< results->idx<<" ; "<<results->score<<std::endl;

    gettimeofday(&detend,NULL);
    double m_dectime = (1000000*(detend.tv_sec - detstart.tv_sec) + detend.tv_usec - detstart.tv_usec);
    std::cout << "Process frames cost "<< m_dectime <<" us" <<std::endl;
   
    return 0;
}
