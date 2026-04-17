#!/usr/bin/env julia
# Find all TIME_WINDOWs under outputs/ and logs/, report their disk usage
# via `du`, and print copy-pasteable bash `rm` commands for the ones not in
# the keep list. This script never modifies the filesystem itself.
#
# Usage:
#   julia scripts/maintenance/prune_time_windows.jl

const KEEP_TIME_WINDOWS = [
    "1958-1987",
    "1989-2018",
    "1968-1977",
    "1999-2008",
    "1972",
    "2003",
]

const SEARCH_ROOTS = ["outputs", "logs/julia", "logs/python", "preprocessed_inputs", "archive"]

is_time_window(name) = occursin(r"^\d{4}(-\d{4})?$", name)

function walk_for_time_windows(root)
    dirs = String[]
    isdir(root) || return dirs
    stack = String[root]
    while !isempty(stack)
        current = pop!(stack)
        for entry in readdir(current; join = true)
            isdir(entry) || continue
            if is_time_window(basename(entry))
                push!(dirs, entry)
            else
                push!(stack, entry)
            end
        end
    end
    return dirs
end

function du_bytes(path)
    out = read(`du -sb $path`, String)
    return parse(Int, split(out)[1])
end

function human(bytes)
    units = ["B", "K", "M", "G", "T", "P"]
    x = float(bytes)
    i = 1
    while x >= 1024 && i < length(units)
        x /= 1024
        i += 1
    end
    return x >= 10 ? string(round(Int, x), units[i]) :
        string(round(x; digits = 1), units[i])
end

project_root = joinpath(@__DIR__, "..", "..")

tw_to_dirs = Dict{String, Vector{String}}()
for sub in SEARCH_ROOTS
    for d in walk_for_time_windows(joinpath(project_root, sub))
        push!(get!(tw_to_dirs, basename(d), String[]), d)
    end
end

sizes = Dict{String, Int}()
print("Measuring disk usage")
for ds in values(tw_to_dirs), d in ds
    print(".")
    sizes[d] = du_bytes(d)
end
println()

println("\n=== Per-TIME_WINDOW totals ===")
keep_total = 0
drop_total = 0
for tw in sort(collect(keys(tw_to_dirs)))
    tag = tw in KEEP_TIME_WINDOWS ? "KEEP" : "DROP"
    total = sum(sizes[d] for d in tw_to_dirs[tw])
    if tw in KEEP_TIME_WINDOWS
        global keep_total += total
    else
        global drop_total += total
    end
    println("$tag  $tw  $(human(total))  ($(length(tw_to_dirs[tw])) dirs)")
    for d in sort(tw_to_dirs[tw])
        println("        $(human(sizes[d]))  $(relpath(d, project_root))")
    end
end

println("\n=== Summary ===")
println("Keep total: $(human(keep_total))")
println("Drop total: $(human(drop_total))")

drop_dirs = String[]
for tw in sort(collect(keys(tw_to_dirs)))
    if !(tw in KEEP_TIME_WINDOWS)
        append!(drop_dirs, sort(tw_to_dirs[tw]))
    end
end

println("\n=== Removal commands (copy-paste to run) ===")
if isempty(drop_dirs)
    println("# (nothing to remove)")
else
    println("rm -rf \\")
    for (i, d) in enumerate(drop_dirs)
        rel = relpath(d, project_root)
        suffix = i == length(drop_dirs) ? "" : " \\"
        println("  $rel$suffix")
    end
end
