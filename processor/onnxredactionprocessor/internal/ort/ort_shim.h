#ifndef ORT_SHIM_H
#define ORT_SHIM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int ort_shim_init(char** out_err);

int ort_shim_create_session(
    const char* model_path,
    int intra_op_threads,
    int concurrent_runs,
    int allow_spinning,
    void** out_env,
    void** out_session,
    void** out_run_options,
    char** out_err
);

void ort_shim_destroy_session(void* env, void* session, void* run_opts);

int ort_shim_terminate(void* run_opts);
int ort_shim_unset_terminate(void* run_opts);
int ort_shim_create_run_options(void** out_run_options, char** out_err);
void ort_shim_release_run_options(void* run_opts);

int ort_shim_run_batch(
    void* session,
    void* run_opts,
    const int64_t* input_ids,
    const int64_t* attention_mask,
    int64_t batch_size,
    int64_t seq_len,
    float* out_logits,
    int64_t out_logits_cap,
    int64_t* out_num_classes,
    char** out_err
);

void ort_shim_free_string(char* s);

#ifdef __cplusplus
}
#endif

#endif // ORT_SHIM_H
