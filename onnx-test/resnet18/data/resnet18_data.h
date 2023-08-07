#include <vector>

// 1 * 3 * 224 * 224
static std::vector<float> test_input{
    #include "data_input.txt"
};

// 1000
static std::vector<float> test_output_gt{
    #include "data_output.txt"
};


