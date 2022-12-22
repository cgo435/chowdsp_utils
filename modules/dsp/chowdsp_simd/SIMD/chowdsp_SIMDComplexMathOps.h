#pragma once

namespace chowdsp::SIMDUtils
{
// @TODO: Remove all these ops once XSIMD has sub-sumed them (hopefully)

using namespace SampleTypeHelpers;

template <typename Type, typename Arch>
inline xsimd::batch<Type, Arch> SIMDComplexMulReal (const xsimd::batch<std::complex<Type>, Arch>& a, const xsimd::batch<std::complex<Type>, Arch>& b)
{
    return (a.real() * b.real()) - (a.imag() * b.imag());
}

template <typename Type, typename Arch>
inline xsimd::batch<Type, Arch> SIMDComplexMulImag (const xsimd::batch<std::complex<Type>, Arch>& a, const xsimd::batch<std::complex<Type>, Arch>& b)
{
    return (a.real() * b.imag()) + (a.imag() * b.real());
}

/** SIMDComplex implementation of std::pow */
template <typename BaseType, typename OtherType, typename Arch>
inline std::enable_if_t<std::is_same_v<NumericType<OtherType>, BaseType>, xsimd::batch<std::complex<BaseType>, Arch>>
    pow (const xsimd::batch<std::complex<BaseType>, Arch>& a, OtherType x)
{
    auto absa = xsimd::abs (a);
    auto arga = xsimd::arg (a);
    auto r = xsimd::pow (absa, xsimd::batch (x));
    auto theta = x * arga;
    auto sincosTheta = xsimd::sincos (theta);
    return { r * sincosTheta.second, r * sincosTheta.first };
}

/** SIMDComplex implementation of std::pow */
template <typename BaseType, typename OtherType, typename Arch>
inline std::enable_if_t<std::is_same_v<NumericType<OtherType>, BaseType>, xsimd::batch<std::complex<BaseType>, Arch>>
    pow (OtherType a, const xsimd::batch<std::complex<BaseType>, Arch>& z)
{
    // same as the complex/complex xsimd implementation, except that we can skip calling arg()!
    const auto ze = xsimd::batch<BaseType, Arch> ((BaseType) 0);

    auto absa = xsimd::abs (a);
    auto arga = xsimd::select (a >= ze, ze, xsimd::batch (juce::MathConstants<BaseType>::pi)); // since a is real, we know arg must be either 0 or pi
    auto x = z.real();
    auto y = z.imag();
    auto r = xsimd::pow (absa, x);
    auto theta = x * arga;

    auto cond = y == ze;
    r = select (cond, r, r * xsimd::exp (-y * arga));
    theta = select (cond, theta, theta + y * xsimd::log (absa));
    auto sincosTheta = xsimd::sincos (theta);
    return { r * sincosTheta.second, r * sincosTheta.first };
}

template <typename BaseType, typename Arch>
inline xsimd::batch<std::complex<BaseType>, Arch> polar (const xsimd::batch<BaseType, Arch>& mag, const xsimd::batch<BaseType, Arch>& angle)
{
    auto sincosTheta = xsimd::sincos (angle);
    return { mag * sincosTheta.second, mag * sincosTheta.first };
}

template <typename BaseType, typename Arch>
inline static xsimd::batch<std::complex<BaseType>, Arch> polar (const xsimd::batch<BaseType, Arch>& angle)
{
    auto sincosTheta = xsimd::sincos (angle);
    return { sincosTheta.second, sincosTheta.first };
}
} // namespace chowdsp::SIMDUtils
