cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(Ultrasound)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/Ultrasound")
set(EXTENSION_CATEGORY "Ultrasound")
set(EXTENSION_CONTRIBUTORS "Tamas Ungi (Queen's University), Tina Kapur (Brigham and Women's Hospital)")
set(EXTENSION_DESCRIPTION "Modules for ultrasound data processing, including anonymization, conversion between data formats and imaging modes, and annotations.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/Ultrasound.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/Ultrasound/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
