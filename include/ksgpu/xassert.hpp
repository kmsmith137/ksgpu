#ifndef _KSGPU_XASSERT_HPP
#define _KSGPU_XASSERT_HPP

#include <string>
#include <sstream>
#include <stdexcept>

// Note: xassert_* macros are implemented with #define, and therefore are outside the ksgpu namespace.

#ifndef _unlikely
#define _unlikely(cond)  (__builtin_expect(cond,0))
#endif


// -------------------------------------------------------------------------------------------------


// xassert(): like assert(), but throws an exception in order to work smoothly with python.
#define xassert(cond) _xassert(cond,__LINE__)
#define _xassert(cond,line) \
    do { \
        if (_unlikely(!(cond))) \
	    throw std::runtime_error("C++ assertion '" __STRING(cond) "' failed (" __FILE__ ":" __STRING(line) ")"); \
    } while (0)

// xassert_msg(): use customized error message.
// The 'msg' argument can either be a (const char *) or a (const std::string &).
#define xassert_msg(cond, msg) \
    do { \
        if (_unlikely(!(cond))) \
	    throw std::runtime_error(msg); \
    } while (0)

// xassert_where(): prepend 'where' string to error message.
// The 'where' argument can either be a (const char *) or a (const sd::string *)
#define xassert_where(cond, where) _xassert_where(cond, where, __LINE__)
#define _xassert_where(cond, where, line) \
    do { \
        if (_unlikely(!(cond))) \
	    throw std::runtime_error(std::string(where) + ": C++ assertion '" __STRING(cond) "' failed (" __FILE__ ":" __STRING(line) ")"); \
    } while (0)


// -------------------------------------------------------------------------------------------------
//
// xassert_eq(), xassert_ne(), xassert_lt(), xassert_le(), xassert_ge(), xassert_gt(), xassert_divisible().
// Compare two arguments, and show their values if the assertion fails.


#define xassert_eq(lhs,rhs) _xassert_eq(lhs,rhs,__LINE__)
#define _xassert_eq(lhs,rhs,line) \
    do { \
	if (_unlikely((lhs) != (rhs))) { \
	    std::stringstream ss; \
	    ss << ("C++ assertion (" __STRING(lhs) ") == (" __STRING(rhs) ") failed (" __FILE__ ":" __STRING(line) "): lhs=") \
	       << lhs << ", rhs=" << rhs; \
	    throw std::runtime_error(ss.str()); \
	} \
    } while (0)

#define xassert_ne(lhs,rhs) _xassert_ne(lhs,rhs,__LINE__)
#define _xassert_ne(lhs,rhs,line) \
    do { \
	if (_unlikely((lhs) == (rhs))) { \
	    std::stringstream ss; \
	    ss << ("C++ assertion (" __STRING(lhs) ") != (" __STRING(rhs) ") failed (" __FILE__ ":" __STRING(line) "): lhs=") \
	       << lhs << ", rhs=" << rhs; \
	    throw std::runtime_error(ss.str()); \
	} \
    } while (0)

#define xassert_lt(lhs,rhs) _xassert_lt(lhs,rhs,__LINE__)
#define _xassert_lt(lhs,rhs,line) \
    do { \
	if (_unlikely((lhs) >= (rhs))) { \
	    std::stringstream ss; \
	    ss << ("C++ assertion (" __STRING(lhs) ") < (" __STRING(rhs) ") failed (" __FILE__ ":" __STRING(line) "): lhs=") \
	       << lhs << ", rhs=" << rhs; \
	    throw std::runtime_error(ss.str()); \
	} \
    } while (0)

#define xassert_le(lhs,rhs) _xassert_le(lhs,rhs,__LINE__)
#define _xassert_le(lhs,rhs,line) \
    do { \
	if (_unlikely((lhs) > (rhs))) { \
	    std::stringstream ss; \
	    ss << ("C++ assertion (" __STRING(lhs) ") <= (" __STRING(rhs) ") failed (" __FILE__ ":" __STRING(line) "): lhs=") \
	       << lhs << ", rhs=" << rhs; \
	    throw std::runtime_error(ss.str()); \
	} \
    } while (0)

#define xassert_gt(lhs,rhs) _xassert_gt(lhs,rhs,__LINE__)
#define _xassert_gt(lhs,rhs,line) \
    do { \
	if (_unlikely((lhs) <= (rhs))) { \
	    std::stringstream ss; \
	    ss << ("C++ assertion (" __STRING(lhs) ") > (" __STRING(rhs) ") failed (" __FILE__ ":" __STRING(line) "): lhs=") \
	       << lhs << ", rhs=" << rhs; \
	    throw std::runtime_error(ss.str()); \
	} \
    } while (0)

#define xassert_ge(lhs,rhs) _xassert_ge(lhs,rhs,__LINE__)
#define _xassert_ge(lhs,rhs,line) \
    do { \
	if (_unlikely((lhs) < (rhs))) { \
	    std::stringstream ss; \
	    ss << ("C++ assertion (" __STRING(lhs) ") >= (" __STRING(rhs) ") failed (" __FILE__ ":" __STRING(line) "): lhs=") \
	       << lhs << ", rhs=" << rhs; \
	    throw std::runtime_error(ss.str()); \
	} \
    } while (0)

#define xassert_divisible(lhs,rhs) _xassert_divisible(lhs,rhs,__LINE__)
#define _xassert_divisible(lhs,rhs,line) \
    do { \
	if (_unlikely((lhs) % (rhs))) { \
	    std::stringstream ss; \
	    ss << ("C++ assertion (" __STRING(lhs) ") % (" __STRING(rhs) ") == 0failed (" __FILE__ ":" __STRING(line) "): lhs=") \
	       << lhs << ", rhs=" << rhs; \
	    throw std::runtime_error(ss.str()); \
	} \
    } while (0)


// -------------------------------------------------------------------------------------------------
//
// xassert_shape_eq(): check that array 'arr' (an object of class ksgpu::Array) has the expected shape.
// If not, throw an exception which shows the actual/expected shapes.
//
// Warning: this macro is fragile -- the expected shape must be written with parentheses and curly braces!
// For example:
//
//   ksgpu::Array<T> arr = ...;
//   xassert_shape_eq(arr, ({3,4,5});   // parentheses and curly braces required!
//
// If the parentheses are omitted, then you'll get a compiler error such as:
//   error: macro "xassert_shape_eq" passed N arguments, but takes just 2
//
// If the curly bracers are omitted, then you'll get a compiler error such as:
//   error: no instance of constructor "std::initializer_list<long>" matches the argument list


#define xassert_shape_eq(arr, expected_shape) \
    _xassert_shape_eq(arr, expected_shape, __LINE__)

#define _xassert_shape_eq(arr, expected_shape, line) \
    do { \
	std::initializer_list<long> s = std::initializer_list<long> expected_shape; \
	if (!ksgpu::_tuples_equal(arr.ndim, arr.shape, s.size(), s.begin())) { \
	    throw std::runtime_error( \
	        "C++ assertion " __STRING(arr) ".shape == " __STRING(expected_shape) " failed (" __FILE__ ":" __STRING(line) "): lhs=" \
		+ ksgpu::_tuple_str(arr.ndim, arr.shape) + ", rhs=" + ksgpu::_tuple_str(s.size(), s.begin()) \
	    ); \
	} \
    } while (0)


#endif // _KSGPU_XASSERT_HPP
