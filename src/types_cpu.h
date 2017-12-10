#pragma once

#include <vector>
#include <complex>
//#include "../intrin/alignedallocator.hpp"

namespace quda {
	using Complex = std::complex<double>;
	using Matrix = std::vector<std::vector<Complex>>;
	using StateVector = std::vector<Complex>;
}
