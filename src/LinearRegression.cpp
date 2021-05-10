//
// Created by Victor Zhang on 7/5/21.
//

#include "LinearRegression.h"

LinearRegression::LinearRegression(int n_feature, int n_target, bool has_bias)
    : n_feature(n_feature), n_target(n_target), has_bias(has_bias),
      weight(n_feature, n_target), bias(1, n_target) {
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
       auto w_ext = x_invsq*(x_ext.t()*target);
       // weight =
   } else {
       auto x_sq = feature.t()*feature;
       auto x_invsq = arma::inv(x_sq);
       weight = x_invsq*(feature.t()*target);
   }
}
