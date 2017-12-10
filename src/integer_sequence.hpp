/// integer_sequence implementation for c++11. 
/// since nvcc 9.0 does not support c++14. inspired by stackoverflow.com/questions/17424477
/// TODO: make join_sequence more compiler cache friendly
#pragma once

#include<type_traits>

#if __cplusplus > 201103L
//	#include <utility>
//	using std::integer_sequence;
//	using std::index_sequence;
//	using std::index_sequence_for;
#else
	template<class T, T... Ints> struct integer_sequence { 
		using type = integer_sequence;
		using value_type = T;
		static constexpr std::size_t size() noexcept {
			return sizeof...(Ints);
		}
	};

	namespace detail{
		template<class T> using Invoke = typename T::type;
		
		//
		template<class T, class S1, class S2> struct join_sequence;
		
		template<class T, T... I1, T... I2> 
		struct join_sequence<T, integer_sequence<T, I1...>, integer_sequence<T, I2...>> 
			: integer_sequence<T, I1..., (sizeof...(I1) + I2)...>
		{};
		
		//
		template<class T, class C> 
		struct make_sequence 
			: join_sequence<T, Invoke<make_sequence<T, std::integral_constant<T, C::value/2>>> 
			                , Invoke<make_sequence<T, std::integral_constant<T, C::value - C::value/2>>>
			                >
		{};
		
		template<class T> struct make_sequence<T, std::integral_constant<T, 0>>
		      : integer_sequence<T>{};
		template<class T> struct make_sequence<T, std::integral_constant<T, 1>>
		      : integer_sequence<T, 0>{};
		
	} // namespace detail

	template<class T, T N> 
	using make_integer_sequence = detail::Invoke<detail::make_sequence<T, std::integral_constant<T, N>>>;
	
	template<std::size_t... Ints>
	using index_sequence = integer_sequence<std::size_t, Ints...>;
	
	template<std::size_t N>
	using make_index_sequence = make_integer_sequence<std::size_t, N>;
	
	template<class... T>
	using index_sequence_for = make_index_sequence<sizeof...(T)>;
#endif // __cplusplus > 201103L



