#include "../include/ksgpu/Dtype.hpp"
#include <sstream>

using namespace std;


namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// Caller will probably want to wrap output with parentheses.
static string dflag_str(unsigned short f)
{
    const char *sep = "";
    static const char *bar = " | ";
    stringstream ss;
    
    if (!f)
	ss << "0";
    if (f & df_int)
	ss << "df_int"; sep = bar;
    if (f & df_uint)
	ss << sep << "df_uint"; sep = bar;
    if (f & df_float)
	ss << sep << "df_float"; sep = bar;
    if (f & df_complex)
	ss << sep << "df_complex"; sep = bar;

    constexpr unsigned short df_all = df_int | df_uint | df_float | df_complex;
    f &= ~df_all;

    if (f)
	ss << sep << "invalid flags 0x" << hex << f;
    return ss.str();
}


// Pretty-print a dtype.
ostream &operator<<(ostream &os, Dtype dt)
{
    if (dt.is_empty()) {
	os << "empty_dtype";
	return os;
    }
    
    if (!dt.is_valid()) {
	os << "invalid_dtype(flags = (" << dflag_str(dt.flags) << "), nbits = " << dt.nbits;
	if (dt.flags == df_complex)
	    os << ", maybe df_complex | df_float was intended?";
	os << ")";
	return os;
    }

    if (dt.flags & df_complex)
	os << ((dt.flags & df_float) ? "complex" : "complex_");
    else if (dt.flags & df_float)
	os << "float";

    if (dt.flags & df_int)
	os << "int";
    else if (dt.flags & df_uint)
	os << "uint";

    if (dt.flags & df_complex)
	os << (dt.nbits/2) << "+" << (dt.nbits/2);
    else
	os << dt.nbits;

    return os;
}


string Dtype::str() const
{
    stringstream ss;
    ss << (*this);
    return ss.str();
}


double Dtype::precision() const
{
    ushort f = (flags & ~df_complex);
    int c = (flags & df_complex) ? 2 : 1;

    if (f == df_float) {
	if (nbits == 64*c) return 1.0e-15;
	if (nbits == 32*c) return 1.0e-6;
	if (nbits == 16*c) return 1.0e-3;
    }
	
    else if ((f == df_int) || (f == df_uint))
	return 0;

    stringstream ss;
    ss << "Dtype::precision(): invalid dtype: " << (*this);
    throw runtime_error(ss.str());
}


Dtype Dtype::real() const
{
    if ((flags & df_complex) == 0)
	return *this;

    if ((nbits & 1) == 0)
	return Dtype(flags & ~df_complex, nbits >> 1);

    stringstream ss;
    ss << "Dtype::real(): invalid dtype: " << (*this);
    throw runtime_error(ss.str());
}


} // namespace ksgpu
