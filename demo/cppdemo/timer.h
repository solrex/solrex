#ifndef _TIMER_H
#define _TIMER_H

#include <sys/time.h>

typedef signed long long time_us_t;

inline time_us_t time_us()
{
    struct timeval now;
    gettimeofday(&now, NULL);
    return now.tv_usec + now.tv_sec * 1000000;
}

#endif
