#pragma once

#include "src/types_cpu.h"

#include <vector>
#include <complex>

using rowtype = quda::StateVector;
using matrixtype = quda::Matrix;

auto randomvector(unsigned size, bool normalize = false)-> rowtype;
auto random_matrix(unsigned size)-> matrixtype;
