set(BITGRAIN_VERSION "2.0.0")

if(NOT TARGET bitgrain::simd)
  add_library(bitgrain::simd SHARED IMPORTED)
  set_target_properties(bitgrain::simd PROPERTIES
    IMPORTED_LOCATION "/tmp/bg_prefix/lib/libbitgrain-simd.so"
    IMPORTED_SONAME "libbitgrain-simd.so.2"
  )
endif()

if(NOT TARGET bitgrain::bitgrain)
  add_library(bitgrain::bitgrain SHARED IMPORTED)
  set_target_properties(bitgrain::bitgrain PROPERTIES
    IMPORTED_LOCATION "/tmp/bg_prefix/lib/libbitgrain.so"
    IMPORTED_SONAME "libbitgrain.so.2"
    INTERFACE_INCLUDE_DIRECTORIES "/tmp/bg_prefix/include/bitgrain"
    INTERFACE_LINK_LIBRARIES "bitgrain::simd;pthread;dl;m"
  )
endif()
