#include <lest/lest.hpp>
#include <approx.hpp>

#include "random_objects.h" 
#include "reference_kernels.hpp"
#include "src/kernels.h"

using lest::ext::approx;

namespace{
	const unsigned n_gate = 1u;   // arity of the gate
	const unsigned ctrlmask = 0u;
}

const lest::test cases[] = {
   CASE("one qubit gate on one qubit system"){
      const unsigned n_system = 1u; // number of qubits in the system
      const auto gate = random_matrix(n_gate);
      
      auto phi_test = randomvector(n_system);
      auto phi_ref = phi_test;
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-10));

      kernel(phi_test, 0u, gate, ctrlmask);
      kernel1test(phi_ref, 0, gate);
      
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
   },
   CASE("one qubit gate on two-qubit system"){
      const unsigned n_system = 2u; // number of qubits in the system
      const auto gate = random_matrix(n_gate);
      
      auto phi_test = randomvector(n_system);
      auto phi_ref = phi_test;
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-10));

      kernel(phi_test, 0u, gate, ctrlmask);
      kernel1test(phi_ref, 0u, gate);
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));

      kernel(phi_test, 1u, gate, ctrlmask);
      kernel1test(phi_ref, 1u, gate);
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
   },
   CASE("one qubit gate on 5-qubit system"){
      const unsigned n_system = 5u; // number of qubits in the system
      const auto gate = random_matrix(n_gate);
      
      auto phi_test = randomvector(n_system);
      auto phi_ref = phi_test;
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-10));
      
      for(unsigned qid = 0; qid < n_system; ++qid){
         kernel(phi_test, qid, gate, ctrlmask);
	      kernel1test(phi_ref, qid, gate);
	      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
      }
   },
};

int main(int argc, char *argv[])
{
	return lest::run(cases, argc, argv);
}
