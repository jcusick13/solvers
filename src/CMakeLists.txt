set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)

add_executable(cg cg.cpp matrix.cpp)
target_compile_options(cg PUBLIC 
    "-Wall"
    "-pedantic")
target_link_libraries(cg ${BLAS_LIBRARIES})
target_include_directories(cg PRIVATE ${BLAS_INCLUDE_DIRS})
