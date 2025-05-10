#pragma once

#ifndef __TEMPLATE_MATRIX_H
#define __TEMPLATE_MATRIX_H

#include "Eigen/Eigen"
#include "gpu_manager_t.h"

typedef double Scalar;

constexpr double default_poisson_ratio = 0.3;
constexpr double default_youngs_modulus = 1e6;


void initTemplateMatrix(Scalar element_len, gpu_manager_t& gm, Scalar ymodu = default_youngs_modulus, Scalar ps_ratio = default_poisson_ratio);

void initTemplateMatrixOrthotropy(Scalar element_len, gpu_manager_t& gm);

const Eigen::Matrix<Scalar, 24, 24>& getTemplateMatrix(void);

const Scalar* getTemplateMatrixElements(void);
const Scalar* getTemplateMatrixElements11(void);
const Scalar* getTemplateMatrixElements12(void);
const Scalar* getTemplateMatrixElements13(void);
const Scalar* getTemplateMatrixElements22(void);
const Scalar* getTemplateMatrixElements23(void);
const Scalar* getTemplateMatrixElements33(void);
const Scalar* getTemplateMatrixElements44(void);
const Scalar* getTemplateMatrixElements55(void);
const Scalar* getTemplateMatrixElements66(void);

Scalar* getDeviceTemplateMatrix(void);


#endif

