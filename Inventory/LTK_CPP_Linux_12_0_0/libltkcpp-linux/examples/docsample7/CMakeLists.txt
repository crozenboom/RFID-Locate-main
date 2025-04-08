#
# This example statically links when possible.  While not required, it is meant
# to illustrate how to handle both static and dynamic linking.  Note that
# producing a fully statically linked application (i.e. linking with '-static')
# will raise a _lot_ of issues, if it works at all, and is not recommended.
#

cmake_minimum_required(VERSION 3.9)
project(docsample7)

find_library(LIBLTKCPP libltkcpp.a REQUIRED)
find_library(LIBLTKCPP_IMPINJ libltkcppimpinj.a REQUIRED)
# it is not recommended to statically link for ssl and crypto libraries
find_library(LIBSSL ssl REQUIRED)
find_library(LIBCRYPTO crypto REQUIRED)
# The ETK does not contain a host static library for xml2.  Add the generic
# name 'xml2' to link against the dynmaic library when compiling for host.
find_library(LIBXML2 NAMES libxml2.a xml2 REQUIRED)
set(LIBS
    ${LIBLTKCPP}
    ${LIBLTKCPP_IMPINJ}
    ${LIBCRYPTO}
    ${LIBSSL}
    ${LIBXML2}
)

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})
