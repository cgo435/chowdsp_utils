setup_catch_test(chowdsp_core_test MODULES chowdsp_core)

target_sources(chowdsp_core_test
    PRIVATE
        DoubleBufferTest.cpp
        AtomicHelpersTest.cpp
        MemoryAliasingTest.cpp
        TupleHelpersTest.cpp
        TypesListTest.cpp
        BindingsTest.cpp
        IteratorsTest.cpp
        OptionalPointerTest.cpp
        StringLiteralTest.cpp
        VectorHelpersTest.cpp
)

target_compile_features(chowdsp_core_test PRIVATE cxx_std_20)
