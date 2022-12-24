#pragma once

namespace chowdsp
{
/** Extensions of juce::FloatVectorOperations */
class FloatVectorOperations
{
public:
    /** Returns true if FloatVectorOperations will be performed using the Apple vDSP framework */
    [[maybe_unused]] static bool isUsingVDSP();

    [[maybe_unused]] static bool isUsingAdvancedSIMDArch();

    [[maybe_unused]] static void setUsingAdvancedSIMDArch (bool canUse);

    /** Divides a vector of values by the src vector. */
    template <typename Arch = void>
    static void divide (float* dest, const float* dividend, const float* divisor, int numValues) noexcept;

    /** Divides a vector of values by the src vector. */
    template <typename Arch = void>
    static void divide (double* dest, const double* dividend, const double* divisor, int numValues) noexcept;

    /** Divides a scalar value by the src vector. */
    template <typename Arch = void>
    static void divide (float* dest, float dividend, const float* divisor, int numValues) noexcept;

    /** Divides a scalar value by the src vector. */
    template <typename Arch = void>
    static void divide (double* dest, double dividend, const double* divisor, int numValues) noexcept;

    /** Sums all the values in the given array. */
    template <typename Arch = void>
    static float accumulate (const float* src, int numValues) noexcept;

    /** Sums all the values in the given array. */
    template <typename Arch = void>
    static double accumulate (const double* src, int numValues) noexcept;

    /** Computes the inner product between the two arrays. */
    template <typename Arch = void>
    static float innerProduct (const float* src1, const float* src2, int numValues) noexcept;

    /** Computes the inner product between the two arrays. */
    template <typename Arch = void>
    static double innerProduct (const double* src1, const double* src2, int numValues) noexcept;

    /** Finds the absolute maximum value in the given array. */
    template <typename Arch = void>
    static float findAbsoluteMaximum (const float* src, int numValues) noexcept;

    /** Finds the absolute maximum value in the given array. */
    template <typename Arch = void>
    static double findAbsoluteMaximum (const double* src, int numValues) noexcept;

    /** Takes the exponent of each value to an integer power. */
    template <typename Arch = void>
    static void integerPower (float* dest, const float* src, int exponent, int numValues) noexcept;

    /** Takes the exponent of each value to an integer power. */
    template <typename Arch = void>
    static void integerPower (double* dest, const double* src, int exponent, int numValues) noexcept;

    /** Computes the Root-Mean-Square average of the input data. */
    template <typename Arch = void>
    static float computeRMS (const float* src, int numValues) noexcept;

    /** Computes the Root-Mean-Square average of the input data. */
    template <typename Arch = void>
    static double computeRMS (const double* src, int numValues) noexcept;

    /** Counts the number of NaN values in the input data */
    [[maybe_unused]] static int countNaNs (const float* src, int numValues) noexcept;

    /** Counts the number of NaN values in the input data */
    [[maybe_unused]] static int countNaNs (const double* src, int numValues) noexcept;

    /** Counts the number of Inf values in the input data */
    [[maybe_unused]] static int countInfs (const float* src, int numValues) noexcept;

    /** Counts the number of Inf values in the input data */
    [[maybe_unused]] static int countInfs (const double* src, int numValues) noexcept;

private:
    FloatVectorOperations() = default; // static use only

    inline static bool canUseAdvancedSIMDArch = false;
};
} // namespace chowdsp
