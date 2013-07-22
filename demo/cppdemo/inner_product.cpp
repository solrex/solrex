#include <cstdlib>
#include <iostream>
#include <list>
#include <numeric>
#include <vector>

#include "timer.h"

using namespace std;

int main()
{
    double a[1000];
    double b[1000];
    vector<double> a_v;
    vector<double> b_v;
    list<double> a_l;
    list<double> b_l;

    double ip[2];
    for (size_t i=0; i<1000; i++) {
        a[i] = random() * 1;
        b[i] = random() * 2;
        a_v.push_back(a[i]);
        a_l.push_back(a[i]);
        b_v.push_back(b[i]);
        b_l.push_back(b[i]);
    }

    time_us_t s, m, e;

    s = time_us();
    for (size_t i=0; i<10000; i++) {
        ip[0] = inner_product(a, a+1000, b, 0.0);
    }
    m = time_us();
    for (size_t i=0; i<10000; i++) {
        ip[1] = 0;
        for (size_t j=0; j<1000; j++) {
            ip[1] += a[j]*b[j];
        }
    }
    e = time_us();
    cout << "a*b = " << ip[0] << "(" << (m-s)/1000.0 << "ms) = " << ip[1]
         << "(" << (e-m)/1000.0 << "ms)" << endl;

    s = time_us();
    for (size_t i=0; i<10000; i++) {
        ip[0] = inner_product(a_v.begin(), a_v.end(), b_v.begin(), 0.0);
    }
    m = time_us();
    for (size_t i=0; i<10000; i++) {
        ip[1] = 0;
        for (size_t j=0; j<1000; j++) {
            ip[1] += a_v[j]*b_v[j];
        }
    }
    e = time_us();
    cout << "a_v*b_v = " << ip[0] << "(" << (m-s)/1000.0 << "ms) = " << ip[1]
         << "(" << (e-m)/1000.0 << "ms)" << endl;

    return 0;
}
