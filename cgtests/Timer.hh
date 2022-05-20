/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   timer.hh
 * Author: dtrad
 *
 * Created on April 9, 2019, 2:46 PM
 */

#ifndef TIMER_HH
#define TIMER_HH
#include <time.h>
#include <sys/time.h>

class timer {
public:
    timer();
    void start();    
    void end();
    void reset();
    int totalTime;
    ~timer();
private:
    struct timeval t1;
    struct timeval t2;
    double seconds;
    
    
};

#endif /* TIMER_HH */

