#ifndef CMINIMAX_H
#define CMINIMAX_H

#include <iostream>
#include <vector>

const float FLOAT_MAX = 1000000.0;
const float FLOAT_MIN = -FLOAT_MAX;

namespace tools {

    class CMinMaxStats {
        public:
            float maximum, minimum;

            CMinMaxStats();
            ~CMinMaxStats();

            void update(float value);
            void clear();
            float normalize(float value);

            float get_max();
            float get_min();
    };

    class CMinMaxStatsList {
        public:
            int num, index;
            std::vector<CMinMaxStats> stats_lst;

            CMinMaxStatsList();
            CMinMaxStatsList(int num);
            ~CMinMaxStatsList();

            float get_max(int index);
            float get_min(int index);
    };
}

#endif