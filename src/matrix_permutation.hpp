#include "types.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <vector>

/// continuous data structure with 2d (matrix-like - row,column, C layout) indexing
template<class T>
struct vec2D : public std::vector<T> {
	using base_t = std::vector<T>;
	
	template<class U>
	vec2D(U&& range, size_t nrows, size_t ncols)
	   : base_t(std::begin(range), std::end(range)), _ncols(ncols)
	{ assert(this->size() == nrows*ncols); }
	
	vec2D(size_t nrows, size_t ncols): vec2D(base_t(nrows*ncols), nrows, ncols) {}
	
	auto operator()(size_t row, size_t col)-> T& { 
		assert(row < nrows());
		assert(col < ncols());
		return (*this)[row*_ncols + col]; 
	}
	auto operator()(size_t row, size_t col) const-> T const& { 
		assert(row < nrows());
		assert(col < ncols());
		return (*this)[row*_ncols + col]; 
	}

	auto nrows() const-> size_t { return this->size()/_ncols; }
	auto ncols() const-> size_t { return _ncols; }

private: // hide parts of vector interface
	using base_t::resize;

private: // data
		size_t _ncols;
}; // struct vec2D


/// doc me
template<size_t N>
struct Offsets_matrix {
	std::array<unsigned, N> offsets;
	vec2D<quda::dev_complex> matrix;
}; // struct Offsets_matrix

/// doc me
template <class M, size_t N>
auto sorted_offsets_and_matrix(const std::array<unsigned, N>& qids ///< range of qubit id-s
                               , M const& matrix                   ///< gate matrix representation
                               )-> Offsets_matrix<N>
{
	using Pair = std::array<unsigned, 2>;

	std::vector<Pair> qubits(qids.size());
	for(unsigned i = 0; i < qubits.size(); ++i){
		qubits[i] = {i, qids[i]};
	}
	std::sort(qubits.begin(), qubits.end()
	          , [](Pair const& p1, Pair const& p2){ return p1[1] < p2[1]; });
	
	auto permuted_matrix = vec2D<quda::dev_complex>(matrix.size(), matrix.size());
	for(unsigned i = 0; i < matrix.size(); ++i){
		for(unsigned j = 0; j < matrix.size(); ++j){
			unsigned old_i = 0, old_j = 0;
			for(unsigned k = 0; k < qubits.size(); ++k){
				old_i |= ((i >> k)&1) << qubits[k][0];
				old_j |= ((j >> k)&1) << qubits[k][0];
			}
			permuted_matrix(i, j) = matrix[old_i][old_j];
		}
	}
	
	auto offsets = std::array<unsigned, N>{};
	std::transform(qubits.rbegin(), qubits.rend(), offsets.begin()
	               , [](Pair p){return 1u << p[1];});
	
	return {offsets, permuted_matrix};
}
