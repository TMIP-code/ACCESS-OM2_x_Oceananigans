"""
    summarize_OU_logs.jl

Group parsed PBS `.OU` logs by (queue, NCPUs, NGPUs, Mem) configuration
and print a summary table with SU/hr rate. Writes the table to `GADI_COSTS.md`.
"""

include("parse_OU_logs.jl")

using Printf

function summarize()
    ou_files = collect_ou_files()

    records = JobRecord[]
    for f in ou_files
        rec = parse_ou_file(f)
        !isnothing(rec) && push!(records, rec)
    end

    # Group by (queue, ncpus, ngpus, mem_requested)
    groups = Dict{Tuple{String, Int, Int, String}, Vector{JobRecord}}()
    for r in records
        _, queue = expected_su_and_queue(r)
        ngpus = isnothing(r.ngpus) ? 0 : r.ngpus
        key = (queue, r.ncpus, ngpus, r.mem_requested)
        push!(get!(groups, key, JobRecord[]), r)
    end

    # Compute SU/hr for each config from the formula
    config_rows = []
    for ((queue, ncpus, ngpus, mem), recs) in groups
        mem_match = match(r"([\d.]+)GB", mem)
        isnothing(mem_match) && continue
        mem_gb = parse(Float64, mem_match[1])

        rate, mem_per_core = if queue == "gpuvolta"
            GPUVOLTA
        elseif queue == "normal"
            NORMAL
        else
            EXPRESS
        end
        charge_basis = max(ncpus, mem_gb / mem_per_core)
        su_per_hr = rate * charge_basis

        total_su = sum(r.service_units for r in recs)
        n = length(recs)

        push!(config_rows, (; queue, ncpus, ngpus, mem, su_per_hr, n, total_su))
    end

    # Sort by queue then ncpus
    sort!(config_rows; by = r -> (r.queue, r.ncpus, r.ngpus))

    # Print to stdout
    @printf(
        "%-9s  %5s  %5s  %10s  %8s  %5s  %10s\n",
        "Queue", "NCPUs", "NGPUs", "Mem Req", "SU/hr", "Jobs", "Total SU",
    )
    println(repeat('-', 62))
    for r in config_rows
        @printf(
            "%-9s  %5d  %5s  %10s  %8.0f  %5d  %10.2f\n",
            r.queue,
            r.ncpus,
            r.ngpus > 0 ? string(r.ngpus) : "",
            r.mem,
            r.su_per_hr,
            r.n,
            r.total_su,
        )
    end

    # Write markdown file
    md_path = joinpath(@__DIR__, "..", "GADI_COSTS.md")
    open(md_path, "w") do io
        println(io, "# Gadi SU Cost Reference")
        println(io)
        println(io, "Regenerate this file with:")
        println(io, "```bash")
        println(io, "julia src/summarize_OU_logs.jl")
        println(io, "```")
        println(io)
        println(io, "## Formula")
        println(io)
        println(io, "```")
        println(io, "SU = Queue_Rate × Max(NCPUs, mem_requested_GB / mem_per_core_GB) × Walltime_hours")
        println(io, "```")
        println(io)
        println(io, "where `mem_per_core = node_RAM / node_CPUs` depends on the node type.")
        println(io)
        println(io, "| Queue | Rate (SU/CPU·hr) | Node type | CPUs/node | RAM/node | Mem/core |")
        println(io, "|-------|-----------------|-----------|-----------|----------|----------|")
        println(io, "| normal | 2 | Cascade Lake | 48 | 192 GiB | 4 GiB |")
        println(io, "| express | 6 | Cascade Lake | 48 | 192 GiB | 4 GiB |")
        println(io, "| gpuvolta | 3 | V100 + Cascade Lake | 48 | 384 GiB | 8 GiB |")
        println(io)
        println(io, "Source: https://opus.nci.org.au/spaces/Help/pages/236880942/Job+Costs")
        println(io)
        println(io, "Formula validated against $(length(records)) `.OU` log files (see `src/parse_OU_logs.jl`).")
        println(io)
        println(io, "## Configurations used and SU rates")
        println(io)
        println(io, "| Queue | NCPUs | NGPUs | Mem | SU/hr | Jobs | Total SU |")
        println(io, "|-------|-------|-------|-----|-------|------|----------|")
        for r in config_rows
            ngpus_str = r.ngpus > 0 ? string(r.ngpus) : "—"
            @printf(
                io,
                "| %s | %d | %s | %s | %d | %d | %.2f |\n",
                r.queue, r.ncpus, ngpus_str, r.mem, r.su_per_hr, r.n, r.total_su,
            )
        end
        println(io)
        @printf(io, "**Total: %d jobs, %.2f SU**\n", length(records), sum(r.service_units for r in records))
    end

    println()
    return println("Wrote $(md_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    summarize()
end
