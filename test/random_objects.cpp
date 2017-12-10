#include "random_objects.h"
#include <random>

namespace {
	
}
///< generate random wavevector
auto randomvector(unsigned nqubits ///< number of qubits in the system
                  , bool normalize)-> rowtype
{
	static auto gen = std::mt19937(42); //  gen(rd());
	std::uniform_real_distribution<> rnd(-1.0, 1.0);

	auto r = rowtype(1u << nqubits);
	for(auto& x : r){ x = {rnd(gen), rnd(gen)}; }
	return r;
}

///< generate random gate matrix
auto random_matrix(unsigned nqubits ///< arity of the gate
                   )-> matrixtype 
{
	const auto size = 1u << nqubits;
	auto m = matrixtype(size);
	for(auto& v : m) { v = randomvector(nqubits); }
	return m;
}
