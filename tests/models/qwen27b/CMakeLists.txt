# Model-specific checks live here; generic Qwen operator controls stay in qwen/.
add_test(NAME qwen27b.tools
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools_test.py)
set_tests_properties(qwen27b.tools PROPERTIES LABELS "cpu;qwen27b;tools")

function(add_qwen27b_test target source labels)
  add_executable(${target} ${CMAKE_CURRENT_SOURCE_DIR}/${source})
  target_link_libraries(${target} PRIVATE gufo_core)
  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target} PROPERTIES LABELS "qwen27b;${labels}"
    SKIP_RETURN_CODE 77)
endfunction()

add_qwen27b_test(qwen27b_vision_test vision_test.cpp "cpu;vision;correctness")

add_qwen27b_test(qwen_mtp_reference_test mtp_reference_test.cpp
  "cpu;mtp;external-model")
if (ENGINE_ENABLE_HIP)
  add_qwen27b_test(qwen_sampling_hip_test sampling_test.hip
    "hip;gpu;sampling;correctness")
  target_link_libraries(qwen_sampling_hip_test PRIVATE hip::host)
  add_test(NAME qwen27b.cli-options COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/cli_test.py $<TARGET_FILE:gufo> --scope options)
  set_tests_properties(qwen27b.cli-options PROPERTIES LABELS "qwen27b;cpu;cli")
  add_test(NAME qwen27b.cli COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/cli_test.py $<TARGET_FILE:gufo>)
  set_tests_properties(qwen27b.cli PROPERTIES
    LABELS "qwen27b;gpu;hip;cli;external-model" SKIP_RETURN_CODE 77)
  add_qwen27b_test(qwen27b_target_test target_test.cpp
    "gpu;hip;target;external-model")
  add_qwen27b_test(qwen_kvarn_error_test kvarn_error_test.cpp
    "gpu;hip;kvarn;attention;opt-c207-kv-quant;external-model")
  target_link_libraries(qwen_kvarn_error_test PRIVATE hip::host)
  add_qwen27b_test(qwen_mtp_gpu_test mtp_test.cpp "gpu;hip;mtp;external-model")
  add_qwen27b_test(qwen_dflash_gpu_test dflash_test.cpp
    "gpu;hip;dflash;dflash2;external-model")
  target_compile_definitions(qwen_dflash_gpu_test PRIVATE
    GUFO_QWEN27B_CORPUS_PATH="${PROJECT_SOURCE_DIR}/docs/models/qwen3.8-27b/artifacts/speculative-corpus.json")
  add_qwen27b_test(qwen_dflash_noncausal_attention_ops_test attention_test.cpp
    "gpu;hip;dflash;dflash2;kernels")
  target_link_libraries(qwen_dflash_noncausal_attention_ops_test PRIVATE hip::host)
  add_library(qwen_serving_test_support STATIC
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_disk_store.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_model_runner.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_generation_scheduler.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/inference_backend.cpp)
  target_link_libraries(qwen_serving_test_support PUBLIC gufo_logging
    gufo_core gufo_deepseek_v4_flash gufo_qwen38_flash_next hip::host)
  add_executable(inference_backend_gpu_test serving_test.cpp)
  target_link_libraries(inference_backend_gpu_test PRIVATE qwen_serving_test_support)
  add_executable(qwen_vision_serving_test ../qwen/vision_serving_test.cpp)
  target_link_libraries(qwen_vision_serving_test PRIVATE qwen_serving_test_support)
  add_test(NAME qwen_vision_serving_test COMMAND qwen_vision_serving_test)
  set_tests_properties(qwen_vision_serving_test PROPERTIES
    LABELS "qwen27b;qwen38-flash-next;hip;gpu;vision;external-model"
    SKIP_RETURN_CODE 77)
  add_test(NAME inference_backend_gpu_test COMMAND inference_backend_gpu_test)
  set_tests_properties(inference_backend_gpu_test PROPERTIES
    LABELS "qwen27b;hip;gpu;server;correctness;external-model"
    SKIP_RETURN_CODE 77)

  # Design D21 / tasks 7.2: the production entry must keep one session's
  # DFlash2 continuation equal to AR over the corpus and across a seal boundary.
  add_test(NAME qwen27b.dflash-kvarn COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/dflash_single_session_test.py $<TARGET_FILE:gufo>)
  set_tests_properties(qwen27b.dflash-kvarn PROPERTIES
    LABELS "qwen27b;gpu;hip;dflash;kvarn;external-model;slow"
    SKIP_RETURN_CODE 77)

  # Design D21 / tasks 7.3: multi-session checks live in their own suite and run
  # only when GUFO_MULTI_SESSION_TESTS is set (the `multi-session` label).
  add_test(NAME qwen27b_multisession_target_test COMMAND qwen27b_target_test)
  set_tests_properties(qwen27b_multisession_target_test PROPERTIES
    LABELS "qwen27b;gpu;hip;target;multi-session;external-model"
    ENVIRONMENT "GUFO_MULTI_SESSION_TESTS=1" SKIP_RETURN_CODE 77)
  add_test(NAME qwen_dflash_multisession_gpu_test COMMAND qwen_dflash_gpu_test)
  set_tests_properties(qwen_dflash_multisession_gpu_test PROPERTIES
    LABELS "qwen27b;gpu;hip;dflash;dflash2;multi-session;external-model"
    ENVIRONMENT "GUFO_MULTI_SESSION_TESTS=1" SKIP_RETURN_CODE 77)
  add_test(NAME inference_backend_multisession_gpu_test
    COMMAND inference_backend_gpu_test)
  set_tests_properties(inference_backend_multisession_gpu_test PROPERTIES
    LABELS "qwen27b;hip;gpu;server;multi-session;external-model"
    ENVIRONMENT "GUFO_MULTI_SESSION_TESTS=1" SKIP_RETURN_CODE 77)

endif()
