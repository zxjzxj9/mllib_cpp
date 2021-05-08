//
// Created by Victor Zhang on 7/5/21.
//

#ifndef MLLIB_CPP_LINEARREGRESSION_H
#define MLLIB_CPP_LINEARREGRESSION_H
#include <armadillo>

class LinearRegression {
public:
    LinearRegression(int n_feature, int n_target);

private:
    arma::mat weight;
    arma::mat bias;
};


#endif //MLLIB_CPP_LINEARREGRESSION_H
