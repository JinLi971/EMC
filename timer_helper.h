#ifndef TIMER_HELPER_H
#define TIMER_HELPER_H
#include <time.h>
#include <sys/time.h>
class timer_helper
{
public:
    timer_helper();
    clock_t * timers;
    void Init_timers(int ntasks);
    double update_time(int rank, clock_t time);
    void update_all(int ntasks, clock_t now);
    unsigned long int gettimenow();
    double update_time(unsigned long int before, unsigned long int now);
};

#endif // TIMER_HELPER_H
