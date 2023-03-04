#include "cminimax.h"

namespace tools {

    CMinMaxStats::CMinMaxStats() {
        this->maximum = FLOAT_MIN;
        this->minimum = FLOAT_MAX;
    }

    CMinMaxStats::~CMinMaxStats() {}

    void CMinMaxStats::update(float value) {
        if(value > this->maximum) {
            this->maximum = value;
        }
        if(value < this->minimum) {
            this->minimum = value;
        }
    }

    void CMinMaxStats::clear() {
        this->maximum = FLOAT_MIN;
        this->minimum = FLOAT_MAX;
    }

    float CMinMaxStats::normalize(float value) {
        float norm_value = value;
        float delta = this->maximum - this->minimum;
        std::cout << "delta: " << delta << std::endl;
        if(delta > 0) {
            norm_value = (norm_value - this->minimum) / delta;
        }
        return norm_value;
    }

    float CMinMaxStats::get_max() {
        float ans;
        ans = this->maximum;
        return ans; 
    }

    float CMinMaxStats::get_min() {
        float ans;
        ans = this->minimum;
        return ans;
    }

    //*********************************************************

    CMinMaxStatsList::CMinMaxStatsList() {
        this->num = 0;
    }

    CMinMaxStatsList::CMinMaxStatsList(int num) {
        this->num = num;
        for(int i = 0; i < num; ++i) {
            this->stats_lst.push_back(CMinMaxStats());
        }
    }

    CMinMaxStatsList::~CMinMaxStatsList() {}

    float CMinMaxStatsList::get_max(int index) {
        float max;
        max = this->stats_lst[index].get_max();
        return max;
    }

    float CMinMaxStatsList::get_min(int index) {
        float min;
        min = this->stats_lst[index].get_min();
        return min;
    }

}