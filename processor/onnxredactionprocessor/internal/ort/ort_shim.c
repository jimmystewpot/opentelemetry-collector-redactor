#include "ort_shim.h"
#include <onnxruntime_c_api.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static const OrtApi* g_ort = NULL;
static OrtEnv* g_env = NULL;
static OrtMemoryInfo* g_mem_info = NULL;

int ort_shim_init(char** out_err) {
    if (!g_ort) {
        const OrtApiBase* base = OrtGetApiBase();
        if (!base) {
            if (out_err) *out_err = strdup("ort_shim: OrtGetApiBase returned NULL");
            return -1;
        }
        g_ort = base->GetApi(ORT_API_VERSION);
        if (!g_ort) {
            if (out_err) *out_err = strdup("ort_shim: GetApi(ORT_API_VERSION) returned NULL");
            return -1;
        }
    }
    if (!g_env) {
        OrtStatus* st = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnxredactor", &g_env);
        if (st) {
            if (out_err) *out_err = strdup(g_ort->GetErrorMessage(st));
            g_ort->ReleaseStatus(st);
            return -1;
        }
    }
    if (!g_mem_info) {
        OrtStatus* st = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &g_mem_info);
        if (st) {
            if (out_err) *out_err = strdup(g_ort->GetErrorMessage(st));
            g_ort->ReleaseStatus(st);
            return -1;
        }
    }
    return 0;
}

int ort_shim_create_session(
    const char* model_path,
    int intra_op_threads,
    int concurrent_runs,
    int allow_spinning,
    void** out_env,
    void** out_session,
    void** out_run_options,
    char** out_err
) {
    if (ort_shim_init(out_err) != 0) {
        return -1;
    }

    OrtSessionOptions* opts = NULL;
    OrtStatus* st = g_ort->CreateSessionOptions(&opts);
    if (st) {
        if (out_err) *out_err = strdup(g_ort->GetErrorMessage(st));
        g_ort->ReleaseStatus(st);
        return -1;
    }

    if (intra_op_threads > 0) {
        OrtStatus* s = g_ort->SetIntraOpNumThreads(opts, intra_op_threads);
        if (s) g_ort->ReleaseStatus(s);
    }
    OrtStatus* s = g_ort->SetInterOpNumThreads(opts, 1);
    if (s) g_ort->ReleaseStatus(s);

    if (!allow_spinning) {
        OrtStatus* sc = g_ort->AddSessionConfigEntry(opts, "session.intra_op.allow_spinning", "0");
        if (sc) g_ort->ReleaseStatus(sc);
    }
    OrtStatus* dm = g_ort->DisableMemPattern(opts);
    if (dm) g_ort->ReleaseStatus(dm);

    OrtSession* session = NULL;
    st = g_ort->CreateSession(g_env, model_path, opts, &session);
    g_ort->ReleaseSessionOptions(opts);
    if (st) {
        if (out_err) *out_err = strdup(g_ort->GetErrorMessage(st));
        g_ort->ReleaseStatus(st);
        return -1;
    }

    OrtRunOptions* run_opts = NULL;
    st = g_ort->CreateRunOptions(&run_opts);
    if (st) {
        g_ort->ReleaseSession(session);
        if (out_err) *out_err = strdup(g_ort->GetErrorMessage(st));
        g_ort->ReleaseStatus(st);
        return -1;
    }

    if (out_env) *out_env = (void*)g_env;
    if (out_session) *out_session = (void*)session;
    if (out_run_options) *out_run_options = (void*)run_opts;
    return 0;
}

void ort_shim_destroy_session(void* env, void* session, void* run_opts) {
    (void)env;
    if (g_ort) {
        if (run_opts) {
            g_ort->ReleaseRunOptions((OrtRunOptions*)run_opts);
        }
        if (session) {
            g_ort->ReleaseSession((OrtSession*)session);
        }
    }
}

int ort_shim_terminate(void* run_opts) {
    if (g_ort && run_opts) {
        OrtStatus* st = g_ort->RunOptionsSetTerminate((OrtRunOptions*)run_opts);
        if (st) {
            g_ort->ReleaseStatus(st);
            return -1;
        }
        return 0;
    }
    return -1;
}

int ort_shim_unset_terminate(void* run_opts) {
    if (g_ort && run_opts) {
        OrtStatus* st = g_ort->RunOptionsUnsetTerminate((OrtRunOptions*)run_opts);
        if (st) {
            g_ort->ReleaseStatus(st);
            return -1;
        }
        return 0;
    }
    return -1;
}

int ort_shim_create_run_options(void** out_run_options, char** out_err) {
    if (ort_shim_init(out_err) != 0) {
        return -1;
    }
    OrtRunOptions* run_opts = NULL;
    OrtStatus* st = g_ort->CreateRunOptions(&run_opts);
    if (st) {
        if (out_err) *out_err = strdup(g_ort->GetErrorMessage(st));
        g_ort->ReleaseStatus(st);
        return -1;
    }
    if (out_run_options) *out_run_options = (void*)run_opts;
    return 0;
}

void ort_shim_release_run_options(void* run_opts) {
    if (g_ort && run_opts) {
        g_ort->ReleaseRunOptions((OrtRunOptions*)run_opts);
    }
}

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
) {
    if (!g_ort || !session || !input_ids || !attention_mask || !out_logits) {
        if (out_err) *out_err = strdup("ort_shim: invalid null arguments passed to run_batch");
        return -1;
    }

    if (!g_mem_info) {
        if (ort_shim_init(out_err) != 0) {
            return -1;
        }
    }

    OrtStatus* st = NULL;
    int64_t in_dims[2] = {batch_size, seq_len};
    size_t in_bytes = (size_t)(batch_size * seq_len * sizeof(int64_t));

    OrtValue* in_tensors[2] = {NULL, NULL};
    st = g_ort->CreateTensorWithDataAsOrtValue(
        g_mem_info,
        (void*)input_ids,
        in_bytes,
        in_dims,
        2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &in_tensors[0]
    );
    if (st) {
        if (out_err) *out_err = strdup(g_ort->GetErrorMessage(st));
        g_ort->ReleaseStatus(st);
        return -1;
    }

    st = g_ort->CreateTensorWithDataAsOrtValue(
        g_mem_info,
        (void*)attention_mask,
        in_bytes,
        in_dims,
        2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &in_tensors[1]
    );
    if (st) {
        if (out_err) *out_err = strdup(g_ort->GetErrorMessage(st));
        g_ort->ReleaseStatus(st);
        g_ort->ReleaseValue(in_tensors[0]);
        return -1;
    }

    const char* input_names[] = {"input_ids", "attention_mask"};
    const char* output_names[] = {"logits"};
    OrtValue* out_tensors[1] = {NULL};

    st = g_ort->Run(
        (OrtSession*)session,
        (const OrtRunOptions*)run_opts,
        input_names,
        (const OrtValue* const*)in_tensors,
        2,
        output_names,
        1,
        out_tensors
    );

    // Release input tensors immediately after Run returns
    g_ort->ReleaseValue(in_tensors[0]);
    g_ort->ReleaseValue(in_tensors[1]);

    if (st) {
        if (out_err) *out_err = strdup(g_ort->GetErrorMessage(st));
        g_ort->ReleaseStatus(st);
        if (out_tensors[0]) {
            g_ort->ReleaseValue(out_tensors[0]);
        }
        return -1;
    }

    if (!out_tensors[0]) {
        if (out_err) *out_err = strdup("ort_shim: run produced null output tensor");
        return -1;
    }

    OrtTensorTypeAndShapeInfo* shape_info = NULL;
    st = g_ort->GetTensorTypeAndShape(out_tensors[0], &shape_info);
    if (st) {
        if (out_err) *out_err = strdup(g_ort->GetErrorMessage(st));
        g_ort->ReleaseStatus(st);
        g_ort->ReleaseValue(out_tensors[0]);
        return -1;
    }

    size_t num_dims = 0;
    st = g_ort->GetDimensionsCount(shape_info, &num_dims);
    if (st) g_ort->ReleaseStatus(st);
    int64_t dims[3] = {0};
    if (num_dims > 3) num_dims = 3;
    st = g_ort->GetDimensions(shape_info, dims, num_dims);
    if (st) g_ort->ReleaseStatus(st);
    g_ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    int64_t num_classes = 19;
    if (num_dims >= 3) {
        num_classes = dims[2];
    }
    if (out_num_classes) {
        *out_num_classes = num_classes;
    }

    int64_t total_elements = batch_size * seq_len * num_classes;
    if (total_elements > out_logits_cap) {
        g_ort->ReleaseValue(out_tensors[0]);
        if (out_err) *out_err = strdup("ort_shim: output logits buffer capacity exceeded");
        return -1;
    }

    float* tensor_data = NULL;
    st = g_ort->GetTensorMutableData(out_tensors[0], (void**)&tensor_data);
    if (st) {
        if (out_err) *out_err = strdup(g_ort->GetErrorMessage(st));
        g_ort->ReleaseStatus(st);
        g_ort->ReleaseValue(out_tensors[0]);
        return -1;
    }

    memcpy(out_logits, tensor_data, (size_t)(total_elements * sizeof(float)));
    g_ort->ReleaseValue(out_tensors[0]);

    return 0;
}

void ort_shim_free_string(char* s) {
    if (s) {
        free(s);
    }
}
