#pragma once

#include <type_traits>
#include <array>
#include <complex>

namespace lest{
	namespace ext{
		template<class T>
		struct has_iterator{
			template<class U> static std::true_type  check(typename U::const_iterator*);
			template<class U> static std::false_type check(...);

			static constexpr bool value = std::is_same<std::true_type, decltype(check<T>(0))>::value; // std::result_of<check<T>(nullptr_t)>::type;
		};

		template<class T, class Enable = void>
		struct Approx;

		template<class T>
		struct Approx<T, typename std::enable_if<has_iterator<T>::value>::type> {
			Approx(const T& values)
				: values{&values}
				, eps_{std::numeric_limits<float>::epsilon()*10} // default value
			{}

			Approx& epsilon(double eps) { eps_ = eps; return *this;}

			bool operator== (const T& other) const {
				using std::begin;
				using std::end;
				auto otherStart = begin(other);
				for(auto it = begin(*values); it != end(*values); ++it, ++otherStart){
					auto me = *it;
					auto other = *otherStart;
					if(std::abs(me - other) > 0.5*eps_*std::abs(me + other)){
						std::cerr << "ext::Approx failed with original value equal to " << other
						          << " and approximate value equal to " << me << "\n";
						return false;
					}
				}
				return true;
			}

			bool operator!=(const T& other){ return !(this == other); }

			friend bool operator== (const T& other, const Approx& ap){ return ap == other; }
			friend bool operator!= (const T& other, const Approx& ap){ return ap != other; }
		private:
			T const * const values;
			double eps_;
		};

		template<class T>
		auto approx(const T& iterable) -> Approx<typename std::enable_if<has_iterator<T>::value, T>::type> {
			return Approx<T>(iterable);
		}

	} //namespace ext
} // namespace lest
