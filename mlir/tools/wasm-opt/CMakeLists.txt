get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LLVM_LINK_COMPONENTS
    Core
    Support
)
if(MLIR_INCLUDE_TESTS)
    set(test_libs
            ${cuda_test_libs}
            MLIRTestFuncToLLVM
            MLIRAffineTransformsTestPasses
            MLIRArithTestPasses
            MLIRArmSMETestPasses
            MLIRBufferizationTestPasses
            MLIRControlFlowTestPasses
            MLIRDLTITestPasses
            MLIRFuncTestPasses
            MLIRGPUTestPasses
            MLIRLinalgTestPasses
            MLIRLoopLikeInterfaceTestPasses
            MLIRMathTestPasses
            MLIRTestMathToVCIX
            MLIRMemRefTestPasses
            MLIRTestMemRefToLLVMWithTransforms
            MLIRMeshTest
            MLIRNVGPUTestPasses
            MLIRSCFTestPasses
            MLIRShapeTestPasses
            MLIRSPIRVTestPasses
            MLIRTensorTestPasses
            MLIRTestAnalysis
            MLIRTestConvertToSPIRV
            MLIRTestDialect
            MLIRTestDynDialect
            MLIRTestIR
            MLIRTestPass
            MLIRTestReducer
            MLIRTestTransforms
            MLIRTilingInterfaceTestPasses
            MLIRTosaTestPasses
            MLIRVectorTestPasses
            MLIRXeGPUTestPasses
            MLIRTestVectorToSPIRV
            MLIRLLVMTestPasses
    )
    set(test_libs ${test_libs}
            MLIRTestPDLL
            MLIRTestTransformDialect
            MLIRTestIRDLToCppDialect
    )

    if (MLIR_ENABLE_PDL_IN_PATTERNMATCH)
        set(test_libs ${test_libs}
                MLIRTestPDLL
                MLIRTestRewrite
        )
    endif()
endif()

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}

        MLIRAffineAnalysis
        MLIRAnalysis
        MLIRCastInterfaces
        MLIRDialect
        MLIROptLib
        MLIRParser
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
        MLIRSupport
        MLIRIR
        MLIRToLLVMIRTranslationRegistration
)

add_mlir_tool(wasm-opt
        wasm-opt.cpp
)
mlir_target_link_libraries(wasm-opt PRIVATE ${LIBS})
target_link_libraries(wasm-opt PRIVATE ${test_libs})
llvm_update_compile_flags(wasm-opt)

mlir_check_all_link_libraries(wasm-opt)