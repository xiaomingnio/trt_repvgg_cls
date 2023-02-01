#include"algeventcls.h"
#include<stdio.h>
#include <chrono>
#include<fstream>
#include<iostream>

using namespace std;

int main(int argc, char** argv)
{
    AlgEventCLS* clsprocessor = new AlgEventCLS();
    // read image
    cout << "read image ..." << endl;
    int totalnum = 0;
    int rightnum = 0;
    double totaltime = 0.0;
    std::string valfile = "../data/test.txt";

    std::ifstream infile;
        std::string a, b;

        infile.open(valfile);

        if (infile)
        {
            while (!infile.eof())
            {
                infile >> a;
                std::cout << a << std::endl;
                cv::Mat images= cv::imread(a);
                totalnum ++;
                PredRes *results = new PredRes();

                auto start=std::chrono::system_clock::now();

                clsprocessor->getEvents(images,results);

                auto end = std::chrono::system_clock::now();
                double m_dectime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;

                totaltime += m_dectime;

                std::cout<<"Event : "<<results->cls<<" ; "<<results->score<<std::endl;

            }
        }
        infile.close();
        std::cout << "Ave Time :" << totaltime/totalnum << " us"<< std::endl;
   
    return 0;
}
