//
// Created by Victor Zhang on 7/5/21.
//

#ifndef MLLIB_CPP_MODELBASE_H
#define MLLIB_CPP_MODELBASE_H
#include <armadillo>

class ModelBase {
public:
    virtual void fit(arma::mat feature, arma::mat target) = 0;
    virtual arma::mat fit_transform(arma::mat feature, arma::mat target) = 0;
};


#endif //MLLIB_CPP_MODELBASE_H
