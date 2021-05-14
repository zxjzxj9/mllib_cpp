//
// Created by Victor Zhang on 7/5/21.
//

#include "LinearRegression.h"

LinearRegression::LinearRegression(int n_feature, int n_target, bool has_bias)
    : n_feature(n_feature), n_target(n_target), has_bias(has_bias),
      weight_(n_feature, n_target), bias_(1, n_target) {
}

// arma::mat LinearRegression::fit_transform(arma::mat feature, arma::mat target) {
//     return arma::mat();
// }

void LinearRegression::fit(arma::mat feature, arma::mat target) {
   // x@W = y  (n_sample x n_feature + 1) @ (n_feature + 1, n_target) = y (n_sample, n_target)
   if(has_bias) {
       int n_sample = target.n_cols;
       auto x_ext = arma::join_cols(feature, arma::ones(n_sample, 1));
       auto x_sq = x_ext.t()*x_ext;
       auto x_invsq = arma::inv(x_sq);
       auto w_ext = (x_invsq*(x_ext.t()*target)).eval();
       weight_ = w_ext.rows(arma::span(0, n_feature));
       bias_ = w_ext.rows(arma::span(n_feature, n_feature + 1));
   } else {
       auto x_sq = feature.t()*feature;
       auto x_invsq = arma::inv(x_sq);
       weight_ = x_invsq * (feature.t() * target);
   }


}

int LinearRegression::getNFeature() const {
    return n_feature;
}

int LinearRegression::getNTarget() const {
    return n_target;
}

const arma::mat &LinearRegression::getWeight() const {
    return weight_;
}

bool LinearRegression::isHasBias() const {
    return has_bias;
}

const arma::mat &LinearRegression::getBias() const {
    return bias_;
}

double LinearRegression::getCorr() const {
    return corr_;
}
