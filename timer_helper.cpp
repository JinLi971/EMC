#include "timer_helper.h"
#include <stdlib.h>
timer_helper::timer_helper()
{
}

void timer_helper::Init_timers(int ntasks){
    timers = (clock_t *) malloc(sizeof(clock_t)*ntasks);
    clock_t now ;
    time(&now);
    for(int i=0;i<ntasks; i++)
        timers[i] = now;
}

double timer_helper::update_time(int rank, clock_t time){
    clock_t newT = timers[rank];
    timers[rank] =time;
    return (double)(time - newT)/(double)CLOCKS_PER_SEC;
}


void timer_helper::update_all(int ntasks, clock_t now){
    for(int i=0;i<ntasks; i++)
        timers[i] = now;
}

unsigned long int timer_helper::gettimenow(){
    struct timeval now;
    gettimeofday(&now,NULL);
    return  now.tv_sec*1000+now.tv_usec/1000;
}

double timer_helper::update_time( unsigned long int before, unsigned long int nowi){
    return  (nowi - before) / 1000.0;
}
