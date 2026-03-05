"""
    parse_OU_logs.jl

Parse all PBS `.OU` log files in `logs/PBS/` and print a summary table
of resource usage.
"""

using Printf

const LOGS_DIRS = [
    joinpath(@__DIR__, "..", "logs", "PBS"),
    joinpath(@__DIR__, "..", "archive", "logs", "PBS"),
]

function parse_time_to_seconds(s::AbstractString)
    parts = split(strip(s), ':')
    h, m, sec = parse(Int, parts[1]), parse(Int, parts[2]), parse(Int, parts[3])
    return h * 3600 + m * 60 + sec
end

function format_time(seconds::Int)
    h, rem = divrem(seconds, 3600)
    m, s = divrem(rem, 60)
    return @sprintf("%02d:%02d:%02d", h, m, s)
end

struct JobRecord
    job_id::String
    exit_status::Int
    service_units::Float64
    ncpus::Int
    ngpus::Union{Int, Nothing}
    mem_requested::String
    cpu_time_used::String
    gpu_utilisation::Union{String, Nothing}
    gpu_mem_used::Union{String, Nothing}
    walltime_used::String
end

function parse_ou_file(filepath::String)
    text = read(filepath, String)

    # Find the resource usage block between === lines
    blocks = split(text, r"={10,}")
    # The resource block is typically the second element (between the two === lines)
    resource_block = nothing
    for block in blocks
        if occursin("Resource Usage", block) || occursin("Job Id:", block)
            resource_block = block
            break
        end
    end
    isnothing(resource_block) && return nothing

    lines = split(resource_block, '\n')

    job_id = ""
    exit_status = 0
    service_units = 0.0
    ncpus = 0
    ngpus = nothing
    mem_requested = ""
    cpu_time_used = ""
    gpu_utilisation = nothing
    gpu_mem_used = nothing
    walltime_used = ""

    for line in lines
        line = strip(line)

        if (m = match(r"Job Id:\s+(\S+)", line)) !== nothing
            job_id = m[1]
        end
        if (m = match(r"Exit Status:\s+(-?\d+)", line)) !== nothing
            exit_status = parse(Int, m[1])
        end
        if (m = match(r"Service Units:\s+([\d.]+)", line)) !== nothing
            service_units = parse(Float64, m[1])
        end
        if (m = match(r"NCPUs Requested:\s+(\d+)", line)) !== nothing
            ncpus = parse(Int, m[1])
        end
        if (m = match(r"CPU Time Used:\s+(\S+)", line)) !== nothing
            cpu_time_used = m[1]
        end
        if (m = match(r"Memory Requested:\s+(\S+)", line)) !== nothing
            mem_requested = m[1]
        end
        if (m = match(r"NGPUs Requested:\s+(\d+)", line)) !== nothing
            ngpus = parse(Int, m[1])
        end
        if (m = match(r"GPU Utilisation:\s+(\S+)", line)) !== nothing
            gpu_utilisation = m[1]
        end
        if (m = match(r"GPU Memory Used:\s+(\S+)", line)) !== nothing
            gpu_mem_used = m[1]
        end
        if (m = match(r"Walltime Used:\s+(\S+)", line)) !== nothing
            walltime_used = m[1]
        end
    end

    isempty(job_id) && return nothing

    return JobRecord(
        job_id, exit_status, service_units,
        ncpus, ngpus, mem_requested, cpu_time_used,
        gpu_utilisation, gpu_mem_used, walltime_used,
    )
end

"""Collect all `.OU` files from `LOGS_DIRS`, sorted by filename."""
function collect_ou_files()
    ou_files = String[]
    for dir in LOGS_DIRS
        isdir(dir) || continue
        append!(ou_files, filter(f -> endswith(f, ".OU"), readdir(dir; join = true)))
    end
    sort!(ou_files; by = basename)
    return ou_files
end

"""Compute SU for a given queue rate and mem_per_core."""
function compute_su(r::JobRecord, rate::Int, mem_per_core::Float64, mem_gb::Float64)
    charge_basis = max(r.ncpus, mem_gb / mem_per_core)
    walltime_hrs = parse_time_to_seconds(r.walltime_used) / 3600.0
    return rate * charge_basis * walltime_hrs
end

# Queue configs: (rate, mem_per_core)
const GPUVOLTA = (3, 8.0) # V100 nodes: 384 GiB / 48 CPUs
const EXPRESS = (6, 4.0)  # Cascade Lake: 192 GiB / 48 CPUs
const NORMAL = (2, 4.0)   # Cascade Lake: 192 GiB / 48 CPUs

"""
Compute expected SU and infer the queue from the NCI formula:
    SU = rate × max(ncpus, mem_GB / mem_per_core_GB) × walltime_hours

Returns `(expected_su, inferred_queue)`.

Queue is inferred as:
  - NGPUs > 0  → gpuvolta (rate=3, mem_per_core=8 GiB)
  - No GPU     → whichever of express (rate=6) or normal (rate=2) gives the
                  closest match to the actual SU reported
"""
function expected_su_and_queue(r::JobRecord)
    mem_match = match(r"([\d.]+)GB", r.mem_requested)
    isnothing(mem_match) && return (NaN, "?")
    mem_gb = parse(Float64, mem_match[1])

    has_gpu = !isnothing(r.ngpus) && r.ngpus > 0
    if has_gpu
        return (compute_su(r, GPUVOLTA..., mem_gb), "gpuvolta")
    else
        su_express = compute_su(r, EXPRESS..., mem_gb)
        su_normal = compute_su(r, NORMAL..., mem_gb)
        if abs(su_express - r.service_units) <= abs(su_normal - r.service_units)
            return (su_express, "express")
        else
            return (su_normal, "normal")
        end
    end
end

function main()
    ou_files = collect_ou_files()

    records = JobRecord[]
    for f in ou_files
        rec = parse_ou_file(f)
        !isnothing(rec) && push!(records, rec)
    end

    # Print header
    @printf(
        "%-28s  %-9s  %5s  %5s  %10s  %12s  %6.2s  %8s  %10s  %12s  %6s  %12s  %4s\n",
        "Job Id", "Queue", "NCPUs", "NGPUs", "Mem Req", "Walltime",
        "SU", "Exp. SU", "Mismatch", "CPU Time", "GPU %", "GPU Mem", "Exit",
    )
    println(repeat('-', 150))

    for r in records
        esu, queue = expected_su_and_queue(r)
        mismatch_pct = r.service_units > 0 ? 100 * abs(esu - r.service_units) / r.service_units : 0.0
        mismatch_str = mismatch_pct > 1.0 ? @sprintf("%.1f%%", mismatch_pct) : ""
        @printf(
            "%-28s  %-9s  %5d  %5s  %10s  %12s  %6.2f  %8.2f  %10s  %12s  %6s  %12s  %4d\n",
            r.job_id,
            queue,
            r.ncpus,
            isnothing(r.ngpus) ? "" : string(r.ngpus),
            r.mem_requested,
            r.walltime_used,
            r.service_units,
            esu,
            mismatch_str,
            r.cpu_time_used,
            isnothing(r.gpu_utilisation) ? "" : r.gpu_utilisation,
            isnothing(r.gpu_mem_used) ? "" : r.gpu_mem_used,
            r.exit_status,
        )
    end

    # Summary
    println()
    println("Total jobs: $(length(records))")
    @printf("Total SU:   %.2f\n", sum(r.service_units for r in records))

    successful = filter(r -> r.exit_status == 0, records)
    return @printf("Successful: %d / %d\n", length(successful), length(records))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
