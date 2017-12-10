#include <lest/lest.hpp>
#include <approx.hpp>

#include "random_objects.h" 
#include "reference_kernels.hpp"
#include "src/kernels.h"

using lest::ext::approx;

const unsigned n_gate = 2u;   // arity of the gate
const unsigned ctrlmask = 0u;

const lest::test cases[] = {
   CASE("2-qubit gate on 2-qubit system"){
      const unsigned n_system = 2u; // number of qubits in the system
      
      const auto gate = random_matrix(n_gate);
      
      auto phi_test = randomvector(n_system);
      auto phi_ref = phi_test;
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-10));

      kernel(phi_test, 0u, 1u, gate, ctrlmask);
      kernel2test(phi_ref, 0u, 1u, gate);
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
   },
   CASE("2-qubit gate on 2-qubit system with matrix inversion"){
      const unsigned n_system = 2u; // number of qubits in the system
      
      const auto gate = random_matrix(n_gate);
      
      auto phi_test = randomvector(n_system);
      auto phi_ref = phi_test;
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-10));

      kernel(phi_test, 1u, 0u, gate, ctrlmask);
      kernel2test(phi_ref, 1u, 0u, gate);
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
   },
   CASE("2-qubit gate on 3-qubit system"){
      const unsigned n_system = 3u; // number of qubits in the system
      
      const auto gate = random_matrix(n_gate);
      
      auto phi_test = randomvector(n_system);
      auto phi_ref = phi_test;
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-10));

      kernel(phi_test, 0u, 1u, gate, ctrlmask);
      kernel2test(phi_ref, 0u, 1u, gate);
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
      
      kernel(phi_test, 0u, 2u, gate, ctrlmask);
      kernel2test(phi_ref, 0u, 2u, gate);
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
      
      kernel(phi_test, 1u, 2u, gate, ctrlmask);
      kernel2test(phi_ref, 1u, 2u, gate);
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
   },
   CASE("2-qubit gate on 3-qubit system with matrix inversion"){
      const unsigned n_system = 3u; // number of qubits in the system
      
      const auto gate = random_matrix(n_gate);
      
      auto phi_test = randomvector(n_system);
      auto phi_ref = phi_test;
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-10));

      kernel(phi_test, 1u, 0u, gate, ctrlmask);
      kernel2test(phi_ref, 1u, 0u, gate);
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
      
      kernel(phi_test, 2u, 0u, gate, ctrlmask);
      kernel2test(phi_ref, 2u, 0u, gate);
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
      
      kernel(phi_test, 2u, 1u, gate, ctrlmask);
      kernel2test(phi_ref, 2u, 1u, gate);
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
   },
   CASE("2-qubit gate on 5-qubit system"){
      const unsigned n_system = 5u; // number of qubits in the system
      
      const auto gate = random_matrix(n_gate);
      
      auto phi_test = randomvector(n_system);
      for(unsigned q1 = 0; q1 < n_system; ++q1){
         for(unsigned q2 = 0; q2 < n_system; ++q2) if(q2 != q1){
            auto phi_ref = phi_test;
            
            kernel(phi_test, q1, q2, gate, ctrlmask);
            kernel2test(phi_ref, q1, q2, gate);
            EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
         }
      }
   },
};

int main(int argc, char *argv[])
{
	return lest::run(cases, argc, argv);
}
