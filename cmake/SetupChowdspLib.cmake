set(CHOWDSP_MODULES_DIR "${PROJECT_SOURCE_DIR}/modules")

if (UNIX AND NOT APPLE)
    # We need to link to pthread explicitly on Linux/GCC
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
endif()

function(setup_chowdsp_lib lib_name)
    set(multiValueArgs MODULES)
    cmake_parse_arguments(CHOWDSPLIB "" "" "${multiValueArgs}" ${ARGN})

    message(STATUS "Setting up ChowDSP Static Lib: ${lib_name}, with modules: ${CHOWDSPLIB_MODULES}")
    add_library(${lib_name} STATIC)

    foreach(module IN LISTS CHOWDSPLIB_MODULES)
        unset(module_path)
        find_path(module_path
            NAMES "${module}.h"
            PATHS "${CHOWDSP_MODULES_DIR}/common" "${CHOWDSP_MODULES_DIR}/dsp"
            PATH_SUFFIXES "${module}"
            NO_CACHE
            REQUIRED
        )


        get_filename_component(module_parent_path ${module_path} DIRECTORY)
        target_include_directories(${lib_name} PUBLIC ${module_parent_path})
#        message(STATUS "Adding module: ${module}, with path ${module_path}, from parent ${module_parent_path}")

        if(EXISTS "${module_path}/${module}.cpp")
            target_sources(${lib_name} PRIVATE "${module_path}/${module}.cpp")
#            message(STATUS "Adding source ${module_path}/${module}.cpp")
        endif()
    endforeach()

    target_compile_definitions(${lib_name}
        PUBLIC
            $<IF:$<CONFIG:DEBUG>,DEBUG=1 _DEBUG=1,NDEBUG=1 _NDEBUG=1>
    )

    if(APPLE)
        target_link_libraries(${lib_name} PUBLIC "-framework Accelerate")
    elseif(UNIX)
        target_link_libraries(${lib_name} PUBLIC Threads::Threads)
    endif()
endfunction(setup_chowdsp_lib)