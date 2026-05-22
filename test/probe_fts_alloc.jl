using Oceananigans
using Oceananigans.OutputReaders: FieldTimeSeries, InMemory, Cyclical, Time,
    interpolating_time_indices, cpu_interpolating_time_indices, memory_index
using JLD2: load

include(joinpath(@__DIR__, "..", "src", "shared_utils", "config.jl"))
include(joinpath(@__DIR__, "..", "src", "shared_utils", "grid.jl"))
include(joinpath(@__DIR__, "..", "src", "shared_utils", "data_loading.jl"))

const grid_file = "preprocessed_inputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/grid.jld2"
const mld_file = "preprocessed_inputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/monthly/mld_monthly.jld2"

@info "Loading grid + MLD FTS on CPU"
grid = load_tripolar_grid(grid_file, CPU())
mld_ts = FieldTimeSeries(
    mld_file, "MLD";
    architecture = CPU(), grid = grid,
    backend = InMemory(), time_indexing = Cyclical()
)
@show mld_ts
@show length(mld_ts.times) mld_ts.times[1] mld_ts.times[end]
@info "typeof(mld_ts.times) = $(typeof(mld_ts.times))"
@info "isa AbstractRange: $(mld_ts.times isa AbstractRange)  (range ⇒ no GPU→CPU copy on every call)"

# Pick a time strictly between two snapshots so n₁ ≠ n₂ (the alloc branch)
t_between = (mld_ts.times[1] + mld_ts.times[2]) / 2
@info "Probing at t = $t_between (between snapshots $(mld_ts.times[1]) and $(mld_ts.times[2]))"

# Verify we're hitting the n₁ ≠ n₂ branch
idx = cpu_interpolating_time_indices(CPU(), mld_ts.times, mld_ts.time_indexing, t_between)
@show idx.first_index idx.second_index idx.fractional_index
@assert idx.first_index != idx.second_index "Pick a time strictly between snapshots"

# Pre-allocate a destination Field for the non-allocating alternative
mld_scratch = similar(mld_ts[1])

# ---- Warmup (JIT) ----
mld_ts[Time(t_between)]
mld_ts[Time(t_between)]

@info "============ fts[Time(t)] (current path) ============"
b = @timed mld_ts[Time(t_between)]
@info "Single call:" allocs = b.bytes time_ms = round(b.time * 1.0e3; digits = 3)

GC.gc()
N = 200
b = @timed for _ in 1:N
    mld_ts[Time(t_between)]
end
@info "$N calls:" total_allocs_MB = round(b.bytes / 1024^2; digits = 1) time_ms = round(b.time * 1.0e3; digits = 2) per_call_KB = round(b.bytes / N / 1024; digits = 1) per_call_μs = round(b.time * 1.0e6 / N; digits = 1)

@info "============ set!(scratch, fts[Time(t)]) (callback pattern) ============"
set!(mld_scratch, mld_ts[Time(t_between)])  # warmup
GC.gc()
b = @timed for _ in 1:N
    set!(mld_scratch, mld_ts[Time(t_between)])
end
@info "$N calls:" total_allocs_MB = round(b.bytes / 1024^2; digits = 1) time_ms = round(b.time * 1.0e3; digits = 2) per_call_KB = round(b.bytes / N / 1024; digits = 1) per_call_μs = round(b.time * 1.0e6 / N; digits = 1)

@info "============ Non-allocating alternative: manual interp into scratch ============"
# Pre-resolve everything that can be reused across calls
function interp_into!(dst, fts, t)
    ñ, n₁, n₂ = interpolating_time_indices(fts.time_indexing, fts.times, t)
    if n₁ == n₂
        parent(dst) .= parent(fts[n₁])
    else
        ψ₁ = fts[n₁]
        ψ₂ = fts[n₂]
        parent(dst) .= (1 - ñ) .* parent(ψ₁) .+ ñ .* parent(ψ₂)
    end
    return dst
end

interp_into!(mld_scratch, mld_ts, t_between)  # warmup
GC.gc()
b = @timed for _ in 1:N
    interp_into!(mld_scratch, mld_ts, t_between)
end
@info "$N calls:" total_allocs_MB = round(b.bytes / 1024^2; digits = 1) time_ms = round(b.time * 1.0e3; digits = 2) per_call_KB = round(b.bytes / N / 1024; digits = 1) per_call_μs = round(b.time * 1.0e6 / N; digits = 1)

@info "============ Agent-recipe: view(parent(fts), …) + memory_index ============"
function interp_into_v2!(dst, fts, t)
    indices = cpu_interpolating_time_indices(architecture(fts), fts.times, fts.time_indexing, t)
    ñ, n₁, n₂ = indices.fractional_index, indices.first_index, indices.second_index
    m₁ = memory_index(fts, n₁)
    m₂ = memory_index(fts, n₂)
    p = parent(fts)
    if n₁ == n₂
        parent(dst) .= view(p, :, :, :, m₁)
    else
        parent(dst) .= ñ .* view(p, :, :, :, m₂) .+ (1 - ñ) .* view(p, :, :, :, m₁)
    end
    return dst
end

interp_into_v2!(mld_scratch, mld_ts, t_between)  # warmup
GC.gc()
b = @timed for _ in 1:N
    interp_into_v2!(mld_scratch, mld_ts, t_between)
end
@info "$N calls:" total_allocs_MB = round(b.bytes / 1024^2; digits = 1) time_ms = round(b.time * 1.0e3; digits = 2) per_call_KB = round(b.bytes / N / 1024; digits = 1) per_call_μs = round(b.time * 1.0e6 / N; digits = 1)

@info "============ Production helper: data_loading.jl::interp_fts! ============"
interp_fts!(mld_scratch, mld_ts, t_between)  # warmup
GC.gc()
b = @timed for _ in 1:N
    interp_fts!(mld_scratch, mld_ts, t_between)
end
@info "$N calls:" total_allocs_MB = round(b.bytes / 1024^2; digits = 1) time_ms = round(b.time * 1.0e3; digits = 2) per_call_KB = round(b.bytes / N / 1024; digits = 1) per_call_μs = round(b.time * 1.0e6 / N; digits = 1)

# Sanity check: both non-allocating paths produce the same result as the original.
ref = Array(parent(mld_scratch))  # snapshot from last interp_into_v2! call
set!(mld_scratch, mld_ts[Time(t_between)])
max_err_orig_vs_v2 = maximum(abs, Array(parent(mld_scratch)) .- ref)
@info "max |v2 − fts[Time(t)]|" max_err_orig_vs_v2

interp_into!(mld_scratch, mld_ts, t_between)
max_err_v1_vs_v2 = maximum(abs, Array(parent(mld_scratch)) .- ref)
@info "max |v1 − v2|" max_err_v1_vs_v2

@info "Probe complete"
