/// first working implementation of 1 qubit kernel. for reference puroposes since kernel_n is a bit obstructed.

namespace {
	constexpr unsigned NQUBITS = 1;                        ///< number of qubits the gate acts
	constexpr unsigned NSTATES = 1u << NQUBITS;            ///< length of wavevec corresponding to number of qubits
	constexpr unsigned GATE_MATRIX_SIZE = NSTATES*NSTATES; ///< num elements in the gate matrix representation

	__constant__ char gate_matrix[sizeof(dev_complex)*GATE_MATRIX_SIZE];

	__global__ void kernel_core(dev_complex phi[], unsigned phi_size
										 , unsigned d0)
	{
		const unsigned i = thread_id_to_offset(threadIdx.x + blockDim.x*blockIdx.x, d0);

		dev_complex* matrix = reinterpret_cast<dev_complex*>(gate_matrix);
		if(i < phi_size){
			assert(i + d0 < phi_size);
			apply_gate_qid(matrix, phi + i, d0);
		}
	}

	__global__ void kernel_ctrl_core(dev_complex phi[], unsigned phi_size
										 , unsigned d0, unsigned ctrlmask)
	{
		const unsigned i = thread_id_to_offset(threadIdx.x + blockDim.x*blockIdx.x, d0);

		dev_complex* matrix = reinterpret_cast<dev_complex*>(gate_matrix);
		if(i < phi_size && (i&ctrlmask) == ctrlmask){
			assert(i + d0 < phi_size);
			apply_gate_qid(matrix, phi + i, d0);
		}
	}
 // namespace


void kernel(quda::StateVector& phi, unsigned id0, quda::Matrix const& m, size_t ctrlmask){
	assert(m.size()*m.size() == GATE_MATRIX_SIZE);

	auto ids = std::array<unsigned, NQUBITS>{id0};
	const auto offs_and_m = sorted_offsets_and_matrix(ids, m);
	RUN_CHECKED(cudaMemcpyToSymbol(gate_matrix, (void*)offs_and_m.matrix.data()
	                              , sizeof(dev_complex)*GATE_MATRIX_SIZE));

	dev_complex* d_phi;
	RUN_CHECKED(cudaMalloc((void**)&d_phi, phi.size()*sizeof(dev_complex)));
	RUN_CHECKED(cudaMemcpy(d_phi, phi.data(), phi.size()*sizeof(dev_complex)
	                       , cudaMemcpyHostToDevice));

	const unsigned n_threads = 512;
	const auto n_blocks = divide_up(unsigned(phi.size()), NSTATES*n_threads);
	const auto& offs = offs_and_m.offsets;
	if(ctrlmask == 0){
		kernel_call(kernel_core, KernelParams1D{n_blocks, n_threads}
		            , d_phi, phi.size(), offs[0]);
	} else {
		kernel_call(kernel_ctrl_core, KernelParams1D{n_blocks, n_threads}
		            , d_phi, phi.size(), offs[0], ctrlmask);
	}

	RUN_CHECKED(cudaMemcpy(phi.data(), d_phi, phi.size()*sizeof(dev_complex)
	                       , cudaMemcpyDeviceToHost));
	RUN_CHECKED(cudaFree(d_phi));
}
