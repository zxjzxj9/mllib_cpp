
set(source ${source}
        main.cpp
        )

add_executable(testLinearRegression STATIC ${source})
target_link_libraries(testLinearRegression ${ARMADILLO_LIBRARIES})
target_link_libraries(testLinearRegression mlcpp)
target_include_directories(testLinearRegression PUBLIC armadillo/include)
