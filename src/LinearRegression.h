//
// Created by Victor Zhang on 7/5/21.
//

#ifndef MLLIB_CPP_LINEARREGRESSION_H
#define MLLIB_CPP_LINEARREGRESSION_H
#include <armadillo>
#include "ModelBase.h"

class LinearRegression: public ModelBase {
public:
    LinearRegression(int n_feature, int n_target, bool has_bias=true);

    void fit(arma::mat feature, arma::mat target) override;

    arma::mat fit_transform(arma::mat feature, arma::mat target) override;

private:
    int n_feature;
    int n_target;
    bool has_bias;
    arma::mat weight_; // weight of lm
    arma::mat bias_; // bias of lm
    double corr_; // correlation factors of lm
};


#endif //MLLIB_CPP_LINEARREGRESSION_H
