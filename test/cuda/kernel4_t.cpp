#include <lest/lest.hpp>
#include <approx.hpp>

#include "random_objects.h" 
#include "reference_kernels.hpp"
#include "src/kernels.h"

using lest::ext::approx;

const unsigned n_gate = 4u;   // arity of the gate
const unsigned ctrlmask = 0u;

const lest::test cases[] = {
   CASE("4-qubit gate on 4-qubit system"){
      const unsigned n_system = 4u; // number of qubits in the system
      
      const auto gate = random_matrix(n_gate);
      
      auto phi_test = randomvector(n_system);
      auto phi_ref = phi_test;
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-10));

      kernel(phi_test, 0u, 1u, 2u, 3u, gate, ctrlmask);
      kernel4test(phi_ref, 0u, 1u, 2u, 3u, gate);
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
   },
   CASE("4-qubit gate on 4-qubit system with (reverse order)"){
      const unsigned n_system = 4u; // number of qubits in the system
      
      const auto gate = random_matrix(n_gate);
      
      auto phi_test = randomvector(n_system);
      auto phi_ref = phi_test;
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-10));

      kernel(phi_test, 3u, 2u, 1u, 0u, gate, ctrlmask);
      kernel4test(phi_ref, 3u, 2u, 1u, 0u, gate);
      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
   },
   CASE("4-qubit gate on 4-qubit system (all orders)"){
      const unsigned n_system = 4u; // number of qubits in the system
      
      const auto gate = random_matrix(n_gate);
      
      auto phi_test = randomvector(n_system);
      
      auto d = std::array<unsigned, n_gate>{0, 1, 2, 3};
      do{
         auto phi_ref = phi_test;
         kernel(phi_test, d[0], d[1], d[2], d[3], gate, ctrlmask);
	      kernel4test(phi_ref, d[0], d[1], d[2], d[3], gate);
	      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
      }while(std::next_permutation(d.begin(), d.end()));
   },
   CASE("4-qubit gate on 9-qubit system ('random' qubits)"){
      const unsigned n_system = 9u; // number of qubits in the system
      
      const auto gate = random_matrix(n_gate);
      
      auto phi_test = randomvector(n_system);
      
      auto d = std::array<unsigned, n_gate>{0, 4, 7, 8};
      do{
         auto phi_ref = phi_test;
         kernel(phi_test, d[0], d[1], d[2], d[3], gate, ctrlmask);
	      kernel4test(phi_ref, d[0], d[1], d[2], d[3], gate);
	      EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
      }while(std::next_permutation(d.begin(), d.end()));

      d = std::array<unsigned, n_gate>{2, 4, 6, 7};
      do{
         auto phi_ref = phi_test;
         kernel(phi_test, d[0], d[1], d[2], d[3], gate, ctrlmask);
	      kernel4test(phi_ref, d[0], d[1], d[2], d[3], gate);
         EXPECT(phi_ref == approx(phi_test).epsilon(1e-8));
      }while(std::next_permutation(d.begin(), d.end()));
   },
};

int main(int argc, char *argv[])
{
	return lest::run(cases, argc, argv);
}
