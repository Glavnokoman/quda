#include "kernel_n.cuh"

void kernel(quda::StateVector& phi, unsigned id0, quda::Matrix const& m, size_t ctrlmask){
	kernel_(phi, m, ctrlmask, id0);
}

void kernel(quda::StateVector& phi, unsigned id1, unsigned id0, quda::Matrix const& m, size_t ctrlmask){
	kernel_(phi, m, ctrlmask, id1, id0);
}

void kernel(quda::StateVector& phi, unsigned id2, unsigned id1, unsigned id0, quda::Matrix const& m, size_t ctrlmask){
	kernel_(phi, m, ctrlmask, id2, id1, id0);
}

void kernel(quda::StateVector& phi, unsigned id3, unsigned id2, unsigned id1, unsigned id0, quda::Matrix const& m, size_t ctrlmask){
	kernel_(phi, m, ctrlmask, id3, id2, id1, id0);
}

void kernel(quda::StateVector& phi, unsigned id4, unsigned id3, unsigned id2, unsigned id1, unsigned id0, quda::Matrix const& m, size_t ctrlmask){
	kernel_(phi, m, ctrlmask, id4, id3, id2, id1, id0);
}

void kernel(quda::StateVector& phi, unsigned id5, unsigned id4, unsigned id3, unsigned id2, unsigned id1, unsigned id0, quda::Matrix const& m, size_t ctrlmask){
	kernel_(phi, m, ctrlmask, id5, id4, id3, id2, id1, id0);
}

