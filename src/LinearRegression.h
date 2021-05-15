//
// Created by Victor Zhang on 7/5/21.
//

#ifndef MLLIB_CPP_LINEARREGRESSION_H
#define MLLIB_CPP_LINEARREGRESSION_H
#include <armadillo>
#include "ModelBase.h"

namespace mllib {
    class LinearRegression : public ModelBase {
    public:
        LinearRegression(int n_feature, int n_target, bool has_bias = true);

        void fit(arma::mat feature, arma::mat target) override;

        arma::mat fit_transform(arma::mat feature, arma::mat target) override;

        int getNFeature() const;

        int getNTarget() const;

        bool isHasBias() const;

        const arma::mat &getWeight() const;

        const arma::mat &getBias() const;

        double getCorr() const;

    private:
        int n_feature;
        int n_target;
        bool has_bias;
        // weight & bias of lm
        arma::mat weight_;
        arma::mat bias_;
        double corr_; // correlation factors of lm
    };
}

#endif //MLLIB_CPP_LINEARREGRESSION_H
