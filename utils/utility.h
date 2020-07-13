#ifndef TESTBED_UTILITY_H
#define TESTBED_UTILITY_H

namespace tbd
{

// Templated basic pair
template <typename T1, typename T2>
struct pair
{
	T1 first;
	T2 second;
	pair()
	{
		first = T1();
		second = T2();
	}
	pair(T1 first_, T2 second_)
	{
		first = first_;
		second = second_;
	}
};

// Predicate for sorting pairs by second
template<typename T1, typename T2>
bool pair_second_larger_than(const pair<T1, T2>& lhs, const pair<T1, T2>& rhs)
{
	return (lhs.second > rhs.second);
}


// Functor for same thing
template<typename T1>
struct pair_second_larger_functor{

	bool operator()(const T1& lhs, const T1& rhs) const
	{
		return (lhs.second > rhs.second);
	}
};

// Functor for same thing
template<typename T1>
struct pair_second_smaller_functor{

	bool operator()(const T1& lhs, const T1& rhs) const
	{
		return (lhs.second < rhs.second);
	}
};

template<typename T>
void swap2(T& a, T& b)
{
	T temp = a;
	a = b;
	b = temp;
}

// Some const correctness would be nice
template<typename T>
struct less_than_op{
	less_than_op() {}
	bool operator()(T x, T y) {return (x < y);}
};

template<typename T>
struct sum_op{
	sum_op() {}
	T operator()(T x, T y) {return (x + y);}
};

// Sum and integer scale (no float non-type template arguments)
template<typename T, int scale>
struct add_scale_op{
	add_scale_op() {}
	T operator()(T x, T y) {return (x + y) * scale;}
};

// Exclusive scan - out of place with stride
// In must be at least of size (size_out * stride)
template<typename T, typename OP>
void exclusive_scan(T* in, T* out, int size_out, OP my_op, T first, int stride = 1)
{
	out[0] = first;

	for(int i = stride, j = 1; j < size_out; i+=stride, j++)
		out[i] = my_op(out[j-1],in[i-stride]);
}

// Exclusive scan - inplace 
template<typename T, typename OP>
void exclusive_scan(T* inout, int size, OP my_op, T first)
{
	T next;
	T previous = inout[0];

	inout[0] = first;
	for(int i = 1; i < size; i++)
	{
		next = inout[i];
		inout[i] = my_op(inout[i-1],previous);
		previous = next;
	}
}


// Reverse from a to b-1
template<typename T>
void reverse(T* a, T* b)
{
	T temp;
	while(--b > a)
	{
		temp = *a;
		*a = *b;
		*b = temp;
		++a;
	}
}

// Rotate right 'n' places
template<typename T>
void inplace_ror(T* arr, int size, int n)
{
	// Error check
	//...
	reverse(arr, arr+size);
	reverse(arr, arr+n);
	reverse(arr+n, arr+size);
}

}

#endif