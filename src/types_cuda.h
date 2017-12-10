#pragma once

#ifndef CUDA_STABBED
	#include <thrust/device_vector.h>
	#include <thrust/complex.h>
#endif

namespace quda {
#ifndef CUDA_STABBED
	using dev_float = double;
	using dev_complex = thrust::complex<dev_float>;
	using dev_statevector = thrust::device_vector<dev_complex>;	
#else
	using dev_float = double;
	using dev_complex = Complex;
	using dev_statevector = std::vector<dev_complex>;
#endif //CUDA_STABBED
}
