cmake_minimum_required(VERSION 3.12)
project(os_ex3)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS -pthread)

add_executable(os_ex3
        MapReduceFramework.h
        MapReduceFramework.cpp
        Barrier.h
        Barrier.cpp
        SampleClient.cpp
#        WordFrequencies.cpp
#        test.cpp
        )