/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   timer.cc
 * Author: dtrad
 * 
 * Created on April 9, 2019, 2:46 PM
 */

#include "Timer.hh"
#include <iostream>
using namespace std;
timer::timer() {
    totalTime=0;    
}
void timer::start(){
    gettimeofday(&t1, NULL);
}
void timer::end(){
    gettimeofday(&t2, NULL);    
    //totalTime += ((t2.tv_sec - t1.tv_sec)* 1000 + (t2.tv_usec - t1.tv_usec) / 1000); // msec
    totalTime += ((t2.tv_sec - t1.tv_sec)* 1000000 + (t2.tv_usec - t1.tv_usec));  // microsec
}
void timer::reset(){
    totalTime=0;
}
timer::~timer() {
    if (0){
        cerr << "destructor" << endl;
        cerr << (t1.tv_sec) << "  " << t2.tv_sec << " diff = " << t2.tv_sec - t1.tv_sec << endl;
        cerr << (t1.tv_usec) << "  " << t2.tv_usec << " diff = " << t2.tv_usec - t1.tv_usec << endl;
    }
}

