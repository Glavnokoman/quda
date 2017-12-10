#include "types.h"

void kernel(quda::StateVector& phi, unsigned id0, quda::Matrix const& m, size_t ctrlmask);
void kernel(quda::StateVector& phi, unsigned id1, unsigned id0, quda::Matrix const& m, size_t ctrlmask);
void kernel(quda::StateVector& phi, unsigned id2, unsigned id1, unsigned id0, quda::Matrix const& m, size_t ctrlmask);
void kernel(quda::StateVector& phi, unsigned id3, unsigned id2, unsigned id1, unsigned id0, quda::Matrix const& m, size_t ctrlmask);
void kernel(quda::StateVector& phi, unsigned id4, unsigned id3, unsigned id2, unsigned id1, unsigned id0, quda::Matrix const& m, size_t ctrlmask);
void kernel(quda::StateVector& phi, unsigned id5, unsigned id4, unsigned id3, unsigned id2, unsigned id1, unsigned id0, quda::Matrix const& m, size_t ctrlmask);
