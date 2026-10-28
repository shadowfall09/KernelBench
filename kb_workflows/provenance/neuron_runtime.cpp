// Development-only NRT interposition; compiled against the installed SDK headers.
#include <nrt/nrt.h>
#include <nrt/nrt_async.h>
#include <nrt/nrt_experimental.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <link.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

static_assert(std::is_integral<nrta_seq_t>::value && sizeof(nrta_seq_t) <= 8,
              "Unsupported native NRT sequence ABI");

namespace {
struct Model {
    uint64_t generation;
    std::shared_ptr<std::vector<unsigned char>> bytes;
};
struct State {
    std::mutex mutex;
    std::map<uintptr_t, Model> models;
    FILE* journal = nullptr;
    std::string directory;
    pid_t pid = getpid();
    uint64_t next = 0, records = 0, errors = 0, bytes = 0;
    bool finished = false;
    State() {
        const char* path = getenv("KB_NEURON_RUNTIME_DIRECTORY");
        if (!path) return;
        directory = path;
        int fd = open((directory + "/journal.jsonl").c_str(),
                      O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (fd < 0) return;
        journal = fdopen(fd, "w");
        if (!journal) { close(fd); return; }
        fprintf(journal, "{\"type\":\"start\",\"pid\":%d,\"protocol\":1}\n", int(pid));
        fflush(journal);
    }
    bool active() {
        if (!journal || finished || getpid() != pid) return false;
        if (records++ >= 200000) { ++errors; return false; }
        return true;
    }
};
State& state() { static State value; return value; }
thread_local unsigned depth = 0;
struct Boundary { Boundary() { ++depth; } ~Boundary() { --depth; } };
template<class F> F symbol(const char* name) {
    void* result = dlsym(RTLD_NEXT, name);
    if (result) return reinterpret_cast<F>(result);
    // Python extensions commonly load their dependencies RTLD_LOCAL. Locate
    // an already loaded libnrt by its actual loader path; never load a new SDK.
    std::vector<std::string> paths;
    dl_iterate_phdr([](struct dl_phdr_info* info, size_t, void* data) {
        const char* path = info->dlpi_name;
        const char* base = strrchr(path, '/');
        base = base ? base + 1 : path;
        if (!strcmp(base, "libnrt.so") || !strncmp(base, "libnrt.so.", 10))
            static_cast<std::vector<std::string>*>(data)->emplace_back(path);
        return 0;
    }, &paths);
    // Multiple loaded NRT versions cannot supply an unambiguous fallback.
    if (paths.size() != 1) return nullptr;
    void* handle = dlopen(paths[0].c_str(), RTLD_NOW | RTLD_NOLOAD);
    if (!handle) return nullptr;
    result = dlsym(handle, name);
    dlclose(handle);
    return reinterpret_cast<F>(result);
}
uintptr_t key(nrt_model_t* model) { return reinterpret_cast<uintptr_t>(model); }

// Retain the exact immutable byte buffer passed to NRT until successful unload.
// Unknown/oversized loads are forwarded unchanged and cannot acquire identity.
struct Snapshot {
    uint64_t id = 0;
    std::shared_ptr<std::vector<unsigned char>> data;
    std::string filename;
};
Snapshot snapshot(const void* bytes, size_t size) {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Snapshot value;
    if (!s.active()) return value;
    value.id = ++s.next;
    if (!bytes || !size || size > 128ull * 1024 * 1024 ||
        s.bytes + size > 512ull * 1024 * 1024 || value.id > 4096) {
        ++s.errors; return value;
    }
    try {
        value.data = std::make_shared<std::vector<unsigned char>>(
            static_cast<const unsigned char*>(bytes),
            static_cast<const unsigned char*>(bytes) + size);
        value.filename = "nrt-load-" + std::to_string(value.id) + ".neff";
        int fd = open((s.directory + "/" + value.filename).c_str(),
                      O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (fd < 0) throw 1;
        size_t offset = 0;
        while (offset < size) {
            ssize_t n = write(fd, value.data->data() + offset, size - offset);
            if (n <= 0) { close(fd); throw 1; }
            offset += size_t(n);
        }
        if (close(fd) != 0) throw 1;
        s.bytes += size;
    } catch (...) { ++s.errors; value.data.reset(); value.filename.clear(); }
    return value;
}
void loaded(const Snapshot& value, NRT_STATUS status, nrt_model_t* model) {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (status == NRT_SUCCESS && model) {
        if (s.models.count(key(model))) ++s.errors;
        s.models[key(model)] = {value.id, value.data};
    }
    if (!s.active()) return;
    fprintf(s.journal, "{\"type\":\"load\",\"id\":%llu,\"model\":%llu,\"status\":%d,"
            "\"artifact\":\"%s\",\"bytes\":%llu}\n",
            (unsigned long long)value.id, (unsigned long long)key(model), int(status),
            value.filename.c_str(), (unsigned long long)(value.data ? value.data->size() : 0));
    fflush(s.journal);
}
}

__attribute__((constructor)) static void initialize() { state(); }

extern "C" NRT_STATUS nrt_load(const void* bytes, size_t size, int32_t start,
                               int32_t count, nrt_model_t** model) {
    auto fn = symbol<decltype(&nrt_load)>("nrt_load");
    if (!fn) return NRT_FAILURE;
    if (depth) return fn(bytes, size, start, count, model);
    Boundary boundary;
    auto value = snapshot(bytes, size);
    auto status = fn(value.data ? value.data->data() : bytes, size, start, count, model);
    loaded(value, status, status == NRT_SUCCESS && model ? *model : nullptr);
    return status;
}
extern "C" NRT_STATUS nrt_load_collectives(const void* bytes, size_t size, int32_t start,
        int32_t count, uint32_t rank, uint32_t world, nrt_model_t** model) {
    auto fn = symbol<decltype(&nrt_load_collectives)>("nrt_load_collectives");
    if (!fn) return NRT_FAILURE;
    if (depth) return fn(bytes, size, start, count, rank, world, model);
    Boundary boundary;
    auto value = snapshot(bytes, size);
    auto status = fn(value.data ? value.data->data() : bytes, size, start, count, rank, world, model);
    loaded(value, status, status == NRT_SUCCESS && model ? *model : nullptr);
    return status;
}
extern "C" NRT_STATUS nrta_execute_schedule(nrt_model_t* model, const nrt_tensor_set_t* input,
        nrt_tensor_set_t* output, int queue, NRT_STATUS* completion, nrta_seq_t* sequence) {
    auto fn = symbol<decltype(&nrta_execute_schedule)>("nrta_execute_schedule");
    if (!fn) return NRT_FAILURE;
    if (depth) return fn(model, input, output, queue, completion, sequence);
    Boundary boundary;
    auto& s = state();
    uint64_t generation = 0;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        auto it = s.models.find(key(model));
        if (it != s.models.end()) generation = it->second.generation;
    }
    auto status = fn(model, input, output, queue, completion, sequence);
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.active()) {
        fprintf(s.journal, "{\"type\":\"execute\",\"model\":%llu,\"generation\":%llu,"
                "\"sequence\":%llu,\"status\":%d}\n", (unsigned long long)key(model),
                (unsigned long long)generation,
                (unsigned long long)(status == NRT_SUCCESS && sequence ? *sequence : 0), int(status));
        fflush(s.journal);
    }
    return status;
}
extern "C" NRT_STATUS nrt_unload(nrt_model_t* model) {
    auto fn = symbol<decltype(&nrt_unload)>("nrt_unload");
    if (!fn) return NRT_FAILURE;
    if (depth) return fn(model);
    Boundary boundary;
    auto status = fn(model);
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.active()) {
        fprintf(s.journal, "{\"type\":\"unload\",\"model\":%llu,\"status\":%d}\n",
                (unsigned long long)key(model), int(status));
        fflush(s.journal);
    }
    if (status == NRT_SUCCESS) s.models.erase(key(model));
    return status;
}
extern "C" int kb_neuron_runtime_finish() {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.journal || s.finished || getpid() != s.pid) return -1;
    fprintf(s.journal, "{\"type\":\"finish\",\"errors\":%llu}\n", (unsigned long long)s.errors);
    fflush(s.journal);
    int status = ferror(s.journal) || fsync(fileno(s.journal));
    s.finished = true;
    return status ? -1 : 0;
}
