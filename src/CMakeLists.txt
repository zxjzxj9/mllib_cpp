# https://cmake.org/cmake/help/latest/module/FindArmadillo.html

set(source ${source}
    ModelBase.cpp
    LinearRegression.cpp
)

add_library(mlcpp STATIC ${source})
target_link_libraries(mlcpp ${ARMADILLO_LIBRARIES})
target_include_directories(mlcpp PUBLIC armadillo/include)
