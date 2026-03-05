"""
    summarize_TM_age_solves.jl

Parse TM age solve logs (both PBS `.OU` and Julia `.log` files) and print
summary tables of the latest run for each (size, solver, matrix_processing)
configuration. Includes HPC resources, solve timing, and success/failure status.

Writes separate markdown files per advection scheme:
`TM_SOLVER_BENCHMARKS_{advection_scheme}.md`.
"""

include("parse_OU_logs.jl")

using Printf

const JULIA_TM_LOGS_DIRS = [
    joinpath(@__DIR__, "..", "logs", "julia", "TM"),
    joinpath(@__DIR__, "..", "archive", "logs", "julia", "TM"),
]

"""Format a number to 2 significant figures as a string."""
function sigfigs2(x)
    x == 0 && return "0"
    d = floor(Int, log10(abs(x))) + 1  # number of digits before decimal
    if d >= 2
        return string(round(Int, x))
    else
        # need decimal places to get 2 sig figs
        decimals = 2 - d
        return string(round(x; sigdigits = 2))
    end
end

"""Compact human-readable time string with 2 significant figures (e.g. "13s", "1.5min")."""
function prettytime(t)
    isnan(t) && return ""
    if t < 1.0e-6
        return sigfigs2(t * 1.0e9) * "ns"
    elseif t < 1.0e-3
        return sigfigs2(t * 1.0e6) * "μs"
    elseif t < 1
        return sigfigs2(t * 1.0e3) * "ms"
    elseif t < 60
        return sigfigs2(t) * "s"
    elseif t < 3600
        return sigfigs2(t / 60) * "min"
    elseif t < 86400
        return sigfigs2(t / 3600) * "hr"
    else
        return sigfigs2(t / 86400) * "d"
    end
end

"""Format a memory string like "353.492 MiB" to 2 significant figures → "350 MiB"."""
function prettymem(s::AbstractString)
    isempty(s) && return ""
    m = match(r"^([\d.]+)\s*(\S+)$", s)
    isnothing(m) && return s
    val = parse(Float64, m[1])
    unit = m[2]
    return sigfigs2(val) * " " * unit
end

struct SolveRecord
    job_id::String
    model_config::String      # e.g. "cgridtransports_wdiagnosed_centered2_AB2"
    size::String              # "coarse" or "full"
    solver::String            # "Pardiso", "ParU", "UMFPACK", "CUDSS"
    matrix_processing::String # "raw", "symfill", "dropzeros", "symdrop"
    success::Bool
    error_msg::String
    # 1st solve timing (includes factorization)
    solve1_time::Float64      # seconds (NaN if unavailable)
    solve1_mem::String        # memory allocations string (e.g. "353.492 MiB")
    solve1_gpu_mem::String    # GPU memory from CUDA.@time (e.g. "1.23 GiB")
    # 2nd solve timing (pure solve, reuses factorization)
    solve2_time::Float64      # seconds (NaN if N/A)
    solve2_mem::String        # memory allocations string
    solve2_gpu_mem::String    # GPU memory from CUDA.@time
    mean_age::Float64         # volume-weighted mean age in years (NaN if failed)
    # PBS resource info
    pbs_exit_status::Int
    service_units::Float64
    ncpus::Int
    ngpus::Int
end

"""
Parse a solve log filename to extract (size, solver, matrix_processing).

New style: `solve_{MC}_{coarse|full}_{SOLVER}_{MPROC}_{JOBID}.gadi-pbs.log`
Old style: `solve_{MC}_{SOLVER}_{LSprec|prec}_{JOBID}.gadi-pbs.log`

Returns `nothing` for old-style (legacy) filenames.
"""
function parse_solve_filename(filepath::String)
    fname = basename(filepath)

    # New style only
    m = match(r"^solve_(.+)_(coarse|full)_(Pardiso|ParU|UMFPACK|CUDSS)_(\w+)_(\d+)\.gadi-pbs\.log$", fname)
    if !isnothing(m)
        return (model_config = m[1], size = m[2], solver = m[3], matrix_processing = m[4], job_id = m[5])
    end

    return nothing
end

"""
Extract CPU memory allocation string from a @time output line.

Handles both formats:
- `Base.@time`:  `(49.01 k allocations: 353.492 MiB, ...)`
- `CUDA.@time`:  `(2.35 M CPU allocations: 116.257 MiB, ...)`
"""
function extract_alloc_mem(line::AbstractString)
    # Try CUDA.@time format first (explicit "CPU" label)
    m = match(r"CPU allocations?:\s+([\d.]+ \S+iB)", line)
    !isnothing(m) && return m[1]
    # Fall back to Base.@time format (no label)
    m = match(r"allocations?:\s+([\d.]+ \S+iB)", line)
    return isnothing(m) ? "" : m[1]
end

"""
Extract GPU memory from a CUDA.@time output line.

Format: `(X.XX k GPU allocations: 1.23 GiB, ...)`
"""
function extract_gpu_mem(line::AbstractString)
    m = match(r"GPU allocations?:\s+([\d.]+ \S+iB)", line)
    return isnothing(m) ? "" : m[1]
end

"""
Extract timing and memory from a @time / CUDA.@time output block.

Returns `(time_seconds, cpu_mem, gpu_mem)`.
The block is everything on the timing line (may span multiple parenthesized groups).
"""
function extract_timing(text::AbstractString, pattern::Regex)
    m = match(pattern, text)
    isnothing(m) && return (NaN, "", "")
    time_s = parse(Float64, m[1])
    rest = length(m.captures) >= 2 ? m[2] : ""
    return (time_s, extract_alloc_mem(rest), extract_gpu_mem(rest))
end

"""Parse a Julia TM solve log file for timing, success, mean age, and memory."""
function parse_solve_log(filepath::String)
    text = read(filepath, String)

    success = occursin("Volume-weighted mean", text)

    error_msg = ""
    if !success
        m = match(r"ERROR: LoadError: (.+?)$"m, text)
        if !isnothing(m)
            msg = m[1]
            error_msg = length(msg) > 60 ? msg[1:60] * "..." : msg
        elseif occursin("ERROR:", text)
            error_msg = "unknown error"
        else
            error_msg = "incomplete"
        end
    end

    # ── New format: "1st solve" and "2nd solve" labels ──
    # CPU: Base.@time "1st solve" ...  → "1st solve: XX.XX seconds (...)"
    # GPU: print("1st solve: "); CUDA.@time ... → "1st solve: XX.XX seconds (...)"
    solve1_time, solve1_mem, solve1_gpu_mem =
        extract_timing(text, r"1st solve:\s+([\d.]+)\s+seconds\s*(.+?)$"m)
    solve2_time, solve2_mem, solve2_gpu_mem =
        extract_timing(text, r"2nd solve:\s+([\d.]+)\s+seconds\s*(.+?)$"m)

    # ── Legacy format: "solve (full|coarsened) age:" ──
    if isnan(solve1_time)
        solve1_time, solve1_mem, solve1_gpu_mem =
            extract_timing(text, r"solve (?:full|coarsened) age:\s+([\d.]+)\s+seconds\s*(.+?)$"m)
    end

    # ── Legacy GPU format: separate factorization + solve lines ──
    if isnan(solve1_time)
        # GPU LU factorization + GPU solve → combine as 1st solve
        ft, fm, fg = extract_timing(text, r"GPU LU factorization:\s+([\d.]+)\s+seconds\s*(.+?)$"m)
        st, sm, sg = extract_timing(text, r"GPU solve:\s+([\d.]+)\s+seconds\s*(.+?)$"m)
        if !isnan(ft)
            solve1_time = ft + (isnan(st) ? 0.0 : st)
            solve1_mem = fm
            solve1_gpu_mem = fg
        end
    end

    # ── Legacy CUDA.@time (unlabeled): detect by @info context ──
    if isnan(solve1_time)
        m = match(r"Computing LU factorization[^\n]*\n\s*([\d.]+)\s+seconds\s*(.+?)$"m, text)
        if !isnothing(m)
            ft = parse(Float64, m[1])
            fm = extract_alloc_mem(m[2])
            fg = extract_gpu_mem(m[2])
            # Also look for the solve line after it
            m2 = match(r"Solving linear system[^\n]*\n\s*([\d.]+)\s+seconds\s*(.+?)$"m, text)
            st = isnothing(m2) ? 0.0 : parse(Float64, m2[1])
            solve1_time = ft + st
            solve1_mem = fm
            solve1_gpu_mem = fg
        end
    end

    # Mean age
    mean_age = NaN
    m = match(r"Volume-weighted mean \w+ steady age:\s+([\d.]+)\s+years", text)
    if !isnothing(m)
        mean_age = parse(Float64, m[1])
    end

    return (;
        success, error_msg,
        solve1_time, solve1_mem, solve1_gpu_mem,
        solve2_time, solve2_mem, solve2_gpu_mem,
        mean_age,
    )
end

"""Collect all solve `.log` files from Julia TM log directories."""
function collect_solve_logs()
    files = String[]
    for dir in JULIA_TM_LOGS_DIRS
        isdir(dir) || continue
        append!(files, filter(f -> startswith(basename(f), "solve_") && endswith(f, ".log"), readdir(dir; join = true)))
    end
    return files
end

"""Find the PBS `.OU` file for a given job ID."""
function find_ou_file(job_id::AbstractString)
    for dir in LOGS_DIRS
        isdir(dir) || continue
        path = joinpath(dir, "$(job_id).gadi-pbs.OU")
        isfile(path) && return path
    end
    return nothing
end

"""Extract the advection scheme from a model_config string like `cgridtransports_wdiagnosed_centered2_AB2`."""
function advection_scheme(mc::AbstractString)
    parts = split(mc, '_')
    return length(parts) >= 3 ? parts[3] : mc
end

function main()
    solve_files = collect_solve_logs()

    # Parse all solve logs and group by case = (model_config, size, solver, matrix_processing)
    cases = Dict{Tuple{String, String, String, String}, Vector{SolveRecord}}()

    for filepath in solve_files
        info = parse_solve_filename(filepath)
        isnothing(info) && continue  # skip legacy filenames

        job_id = info.job_id
        log_data = parse_solve_log(filepath)

        # Find matching PBS .OU file
        pbs_exit_status = -1
        service_units = 0.0
        ncpus = 0
        ngpus = 0

        ou_path = find_ou_file(job_id)
        if !isnothing(ou_path)
            pbs_rec = parse_ou_file(ou_path)
            if !isnothing(pbs_rec)
                pbs_exit_status = pbs_rec.exit_status
                service_units = pbs_rec.service_units
                ncpus = pbs_rec.ncpus
                ngpus = isnothing(pbs_rec.ngpus) ? 0 : pbs_rec.ngpus
            end
        end

        rec = SolveRecord(
            job_id, info.model_config, info.size, info.solver, info.matrix_processing,
            log_data.success, log_data.error_msg,
            log_data.solve1_time, log_data.solve1_mem, log_data.solve1_gpu_mem,
            log_data.solve2_time, log_data.solve2_mem, log_data.solve2_gpu_mem,
            log_data.mean_age,
            pbs_exit_status, service_units, ncpus, ngpus,
        )

        key = (info.model_config, info.size, info.solver, info.matrix_processing)
        push!(get!(cases, key, SolveRecord[]), rec)
    end

    # Keep only the latest run (highest job ID) per case
    latest = SolveRecord[]
    for (_, recs) in cases
        sort!(recs; by = r -> parse(Int, r.job_id), rev = true)
        push!(latest, first(recs))
    end

    # Group by model_config first
    by_mc = Dict{String, Vector{SolveRecord}}()
    for r in latest
        push!(get!(by_mc, r.model_config, SolveRecord[]), r)
    end

    # Sort model configs: centered2 first, then alphabetically
    mc_list = sort(collect(keys(by_mc)); by = mc -> (advection_scheme(mc) == "centered2" ? 0 : 1, mc))

    matproc_order = Dict("raw" => 0, "symfill" => 1, "dropzeros" => 2, "symdrop" => 3)
    size_order = Dict("full" => 0, "coarse" => 1)

    md_paths = String[]

    for mc in mc_list
        mc_recs = by_mc[mc]
        as = advection_scheme(mc)

        # Group by (matrix_processing, size) for separate tables
        groups = Dict{Tuple{String, String}, Vector{SolveRecord}}()
        for r in mc_recs
            key = (r.matrix_processing, r.size)
            push!(get!(groups, key, SolveRecord[]), r)
        end

        sorted_keys = sort(collect(keys(groups)); by = k -> (get(matproc_order, k[1], 99), get(size_order, k[2], 99)))

        # Print tables to stdout
        println()
        println("# $mc")
        for key in sorted_keys
            recs = sort(groups[key]; by = r -> isnan(r.solve1_time) ? Inf : r.solve1_time)
            matproc, sz = key
            println()
            println("## $sz / $matproc")
            println()
            print_table(stdout, recs)
        end

        # Summary for this model config
        successful = count(r -> r.success, mc_recs)
        @printf(
            "\nTotal: %d cases (%d successful), %.2f SU\n",
            length(mc_recs), successful, sum(r.service_units for r in mc_recs)
        )

        # Write markdown file per advection scheme
        md_path = joinpath(@__DIR__, "..", "TM_SOLVER_BENCHMARKS_$(as).md")
        push!(md_paths, md_path)
        open(md_path, "w") do io
            println(io, "# Transport Matrix Solver Benchmarks — $(as)")
            println(io)
            println(io, "model_config: `$(mc)`")
            println(io)
            println(io, "Regenerate this file with:")
            println(io, "```bash")
            println(io, "julia src/summarize_TM_age_solves.jl")
            println(io, "```")
            println(io)
            println(io, "Latest run per (Size, Solver, MatrixProcessing) configuration.")
            println(io)
            for key in sorted_keys
                recs = sort(groups[key]; by = r -> isnan(r.solve1_time) ? Inf : r.solve1_time)
                matproc, sz = key
                println(io, "## $sz / $matproc")
                println(io)
                print_table_md(io, recs)
                println(io)
            end
            @printf(
                io,
                "**Total: %d cases (%d successful), %.2f SU**\n",
                length(mc_recs), successful, sum(r.service_units for r in mc_recs),
            )
        end
    end

    # Overall summary
    println()
    successful = count(r -> r.success, latest)
    @printf(
        "Grand total: %d cases (%d successful), %.2f SU\n",
        length(latest), successful, sum(r.service_units for r in latest)
    )
    for p in md_paths
        println("Wrote $(p)")
    end
    return
end

"""Print a table to stdout."""
function print_table(io::IO, recs::Vector{SolveRecord})
    @printf(
        io,
        "%-8s  %-6s  %16s  %12s  %12s  %16s  %12s  %12s  %5s  %5s  %6s  %-12s\n",
        "Solver", "Status",
        "1st solve", "1st mem", "1st GPU",
        "2nd solve", "2nd mem", "2nd GPU",
        "CPUs", "GPUs", "SU", "Job ID",
    )
    println(io, repeat('-', 140))
    for r in recs
        @printf(
            io,
            "%-8s  %-6s  %16s  %12s  %12s  %16s  %12s  %12s  %5d  %5s  %6.2f  %-12s\n",
            r.solver,
            r.success ? "OK" : "FAIL",
            prettytime(r.solve1_time),
            prettymem(r.solve1_mem),
            prettymem(r.solve1_gpu_mem),
            prettytime(r.solve2_time),
            prettymem(r.solve2_mem),
            prettymem(r.solve2_gpu_mem),
            r.ncpus,
            r.ngpus > 0 ? string(r.ngpus) : "",
            r.service_units,
            r.job_id,
        )
    end
    return
end

"""Print a markdown table."""
function print_table_md(io::IO, recs::Vector{SolveRecord})
    println(
        io,
        "| Solver | Status | 1st solve | 1st mem | 1st GPU mem | 2nd solve | 2nd mem | 2nd GPU mem | CPUs | GPUs | SU | Job ID |",
    )
    println(
        io,
        "|--------|--------|-----------|---------|-------------|-----------|---------|-------------|------|------|----|--------|",
    )
    for r in recs
        @printf(
            io,
            "| %s | %s | %s | %s | %s | %s | %s | %s | %d | %s | %.2f | %s |\n",
            r.solver,
            r.success ? "OK" : "FAIL",
            prettytime(r.solve1_time),
            prettymem(r.solve1_mem),
            prettymem(r.solve1_gpu_mem),
            prettytime(r.solve2_time),
            prettymem(r.solve2_mem),
            prettymem(r.solve2_gpu_mem),
            r.ncpus,
            r.ngpus > 0 ? string(r.ngpus) : "",
            r.service_units,
            r.job_id,
        )
    end
    return
end

main()
