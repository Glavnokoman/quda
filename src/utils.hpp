#include <array>
#include <cassert>
#include <cstddef>

#ifdef __CUDA_ARCH__
	#define HOST_DEVICE __host__ __device__
#else
	#define HOST_DEVICE
#endif // __CUDA_ARCH__

namespace quda {
	template<class T>
	HOST_DEVICE T thread_to_index(T tid){
		return tid;
	}
	
	template <class T, typename... Args>
	HOST_DEVICE T thread_to_index(T tid, T ext_head, Args... extents){
		const auto index_capacity = ext_head >> (1ul + sizeof...(extents)); // how many threads correspond to unit index increase. assert equal to product of extents of underlying indexes.
		const auto block_id = tid / index_capacity;
		return block_id*ext_head + thread_to_index(tid%index_capacity, extents...);
	}

	template<class T>
	HOST_DEVICE T thread_id_to_offset(T tid){
		return tid;
	}
	
	template<class T, class... Args>
	HOST_DEVICE T thread_id_to_offset(T tid, T d, Args... ds){
		const auto level = sizeof...(ds);
		const auto threads_in_d =  d/(T(1) << level); // how many threads in takes to cover [0-d) offsets
		const auto block = tid/threads_in_d;
		const auto rem = tid%threads_in_d;
		return block*2*d + thread_id_to_offset(rem, ds...);
	}

	/// vector dot product unroll
	template<class T, std::size_t ROW_LENGTH>
	struct Vvm {
		HOST_DEVICE T operator() (T const* v1, T const* v2) {
			return Vvm<T, 1>{}(v1, v2) + Vvm<T, ROW_LENGTH-1>{}(v1 + 1, v2 + 1);
		}
	}; // struct Vvm

	// vector-vector multiplication base case
	template<class T>
	struct Vvm<T, 1ul> {
		HOST_DEVICE T operator() (T const* v1, T const* v2) {
			return (*v1)*(*v2);
		}
	}; // struct Vvm<T, 1>
	
	// base case of matrix-vector multiplication: do nothing
	template<class T, std::size_t ROW_LENGTH, class...>
	HOST_DEVICE void mvm(T const* /*matrix*/, T const*/*vec_src*/, T* /*dst*/){ }
	
	/// compile-time unrolled matrix-vector mutliplication into (permuted) vector
	template<class T, std::size_t ROW_LENGTH, class I, class... Is>
	HOST_DEVICE void mvm (T const* matrix, T const* vec_src, T* dst, I id0, Is... ids) {
		dst[id0] = Vvm<T, ROW_LENGTH>{}(matrix, vec_src);
		mvm<T, ROW_LENGTH>(matrix + ROW_LENGTH, vec_src, dst, ids...);
	}
	
	/// apply gate to vector defined by base pointer and offsets
	template<class T, class... Is>
	HOST_DEVICE void apply_gate_offsets(T const* matrix, T* wavevec, Is... ids){
		constexpr std::size_t N = sizeof...(ids);
		T x[N] = {wavevec[ids]...};
		mvm<T, N>(matrix, x, wavevec, ids...);
	}

	
	template<class...> struct Typelist{};
	template<class P, class...> struct apply_gate_impl{};
	
	// base case
	template<class P, class... Is>
	struct apply_gate_impl<P, Typelist<>, Typelist<Is...>>{
		HOST_DEVICE auto operator()(P const* m, P* v, Is... is)-> void {
			apply_gate_offsets(m, v, is...);
		}
	};
	
	/// Unroll qubit indexes into sequence of offsets in the wave function corresponding to 
	/// flipping those qubits. Resulting sequence length is 2^(num of qubit indexes)
	template<class P, class T, class... Ts, class... Is>
	struct apply_gate_impl<P, Typelist<T, Ts...>, Typelist<Is...>>{
		HOST_DEVICE auto operator()(P const* m, P* v, Ts... ts, T t, Is... is)-> void {
			apply_gate_impl<P, Typelist<Ts...>, Typelist<Is..., Is...>>{}
			    (m, v, ts..., is..., (is + t)...);
		}
	};
	
	/// apply gate to qubit offsets
	template<class P, class... Ts>
	HOST_DEVICE void apply_gate_qid(P const* m, P* v, Ts... ts){
		apply_gate_impl<P, Typelist<Ts...>, Typelist<unsigned>>{}(m, v, ts..., 0u);
	}
}
