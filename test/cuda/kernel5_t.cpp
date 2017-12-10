#include <lest/lest.hpp>
#include <approx.hpp>

#include "random_objects.h" 
#include "reference_kernels.hpp"
#include "src/kernels.h"

using lest::ext::approx;

const unsigned n_gate = 5u;   // arity of the gate
const unsigned ctrlmask = 0u;

const lest::test cases[] = {
   CASE("5-qubit gate on 5-qubit system"){
      const unsigned n_system = 5u; // number of qubits in the system
      
      const auto gate = random_matrix(n_gate);
      
      auto phi_test = randomvector(n_system);
      auto phi_ref = phi_test;
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-10));

      kernel(phi_test, 0u, 1u, 2u, 3u, 4u, gate, ctrlmask);
      kernel5test(phi_ref, 0u, 1u, 2u, 3u, 4u, gate);
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
   },
   CASE("5-qubit gate on 5-qubit system (reverse order)"){
      const unsigned n_system = 5u; // number of qubits in the system
      
      const auto gate = random_matrix(n_gate);
      
      auto phi_test = randomvector(n_system);
      auto phi_ref = phi_test;
 
      kernel(phi_test, 4u, 3u, 2u, 1u, 0u, gate, ctrlmask);
      kernel5test(phi_ref, 4u, 3u, 2u, 1u, 0u, gate);
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
   },
   CASE("5-qubit gate on 5-qubit system ('random' order)"){
      const unsigned n_system = 5u; // number of qubits in the system
      
      const auto gate = random_matrix(n_gate);
      
      auto phi_test = randomvector(n_system);
      auto phi_ref = phi_test;
      
      kernel(phi_test, 4u, 1u, 3u, 2u, 0u, gate, ctrlmask);
      kernel5test(phi_ref, 4u, 1u, 3u, 2u, 0u, gate);
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
      
      kernel(phi_test, 3u, 1u, 4u, 0u, 2u, gate, ctrlmask);
      kernel5test(phi_ref, 3u, 1u, 4u, 0u, 2u, gate);
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
   },
   CASE("5-qubit gate on 11-qubit system ('random' qubits)"){
      const unsigned n_system = 11u; // number of qubits in the system
      
      const auto gate = random_matrix(n_gate);
      
      auto phi_test = randomvector(n_system);
      
      auto d = std::array<unsigned, 5>{0, 3, 4, 8, 10};
      do{
         auto phi_ref = phi_test;
         kernel(phi_test, d[0], d[1], d[2], d[3], d[4], gate, ctrlmask);
	      kernel5test(phi_ref, d[0], d[1], d[2], d[3], d[4], gate);
	      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
      }while(std::next_permutation(d.begin(), d.end()));
   },
};

int main(int argc, char *argv[])
{
	return lest::run(cases, argc, argv);
}
