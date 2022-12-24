setup_catch_test(chowdsp_math_test MODULES chowdsp_math)

target_sources(chowdsp_math_test
    PRIVATE
        CombinatoricsTest.cpp
        FloatVectorOperationsTest.cpp
        MatrixOpsTest.cpp
        PolynomialsTest.cpp
        PowerTest.cpp
        OtherMathOpsTest.cpp
        JacobiEllipticTest.cpp
        PolylogarithmTest.cpp
        RatioTest.cpp
)

if(CHOWDSP_BUILD_TESTS_WITH_AVX)
    chowdsp_add_avx_flags_if_available(chowdsp_math_test)
endif()
